"""
| File: ardupilot_launch_tool.py
| Author: Tomer Tip (tomerT1212@gmail.com)
| Description: Defines an auxiliary tool to launch the Ardupilot process in the background
| License: BSD-3-Clause. Copyright (c) 2024, Tomer Tip. All rights reserved.
"""

# System tools used to launch the ardupilot process in the brackground
import os
import sys
import signal
import tempfile
import subprocess
import psutil


# SITL binary built for each sim_vehicle.py -v vehicle (build/sitl/bin/<name>)
SITL_BINARIES = {
    "ArduCopter": "arducopter",
    "ArduPlane": "arduplane",
    "Rover": "ardurover",
    "ArduSub": "ardusub",
}


def sitl_binary(vehicle):
    """Name of the SITL binary of an ArduPilot vehicle (e.g. "ArduPlane" -> "arduplane")."""
    return SITL_BINARIES.get(vehicle, vehicle.lower())


# Variables set by Isaac Sim's python.sh that must not leak into the SITL processes: they make the
# system python3 (gnome-terminal, sim_vehicle.py, MAVProxy) load Isaac's Python 3.11 stdlib and libcarb.
_ISAAC_ONLY_VARIABLES = ("PYTHONHOME", "LD_PRELOAD")
_ISAAC_PATH_LISTS = ("PYTHONPATH", "LD_LIBRARY_PATH")


def _isaac_roots(environment):
    """Directories owned by the Isaac Sim install running this process."""
    roots = {os.path.realpath(sys.prefix)}
    isaac_path = environment.get("ISAAC_PATH")
    if isaac_path:
        roots.add(os.path.realpath(isaac_path))
    return roots


def _inside(path, roots):
    real = os.path.realpath(path) if path else ""
    return any(real == root or real.startswith(root + os.sep) for root in roots)


def clean_sitl_environment(environment=None):
    """Copy of the environment without Isaac Sim's Python and library paths.

    Entries of PYTHONPATH and LD_LIBRARY_PATH outside Isaac (e.g. ROS) are kept.
    """
    environment = dict(os.environ if environment is None else environment)
    roots = _isaac_roots(environment)
    for name in _ISAAC_ONLY_VARIABLES:
        environment.pop(name, None)
    for name in _ISAAC_PATH_LISTS:
        entries = [entry for entry in environment.get(name, "").split(os.pathsep) if entry and not _inside(entry, roots)]
        if entries:
            environment[name] = os.pathsep.join(entries)
        else:
            environment.pop(name, None)
    return environment


SITL_WRAPPER = """#!/bin/sh
"$@" > "{log}" 2>&1 < /dev/null &
sitl=$!
leader=$(ps -o pgid= -p $$ | tr -d ' ')
while kill -0 {isaac} 2>/dev/null && kill -0 "$leader" 2>/dev/null && kill -0 "$sitl" 2>/dev/null; do
    sleep 1
done
pkill -9 -P "$sitl" 2>/dev/null
kill -9 "$sitl" 2>/dev/null
"""


class ArduPilotLaunchTool:
    """
    A class that manages the start/stop of a ardupilot process. It requires only the path to the Ardupilot installation (assuming that
    Ardupilot was already built with 'make ardupilot_sitl_default none'), the vehicle id and the vehicle model. 
    """

    def __init__(self, ardupilot_dir, vehicle_id: int = 0, ardupilot_model: str = "gazebo-iris",
                 vehicle: str = "ArduCopter"):
        """Construct the ArduPilotLaunchTool object

        Args:
            ardupilot_dir (str): A string with the path to the Ardupilot-Autopilot directory
            vehicle_id (int): The ID of the vehicle. Defaults to 0.
            ardupilot_model (str): The vehicle model. Defaults to "iris".
            vehicle (str): The ArduPilot vehicle for sim_vehicle.py -v ("ArduCopter", "ArduPlane", ...).
                Defaults to "ArduCopter".
        """

        # Attribute that will hold the ardupilot process once it is running
        self.ardupilot_process = None

        # The vehicle id (used for the mavlink port open in the system)
        self.vehicle_id = vehicle_id

        # Configurations to whether autostart ardupilot (SITL) automatically or have the user launch it manually on another
        # terminal
        self.ardupilot_dir = ardupilot_dir

        # Ardupilot frame
        self.ardupilot_model = ardupilot_model

        # ArduPilot vehicle (sim_vehicle.py -v) and the SITL binary it builds
        self.vehicle = vehicle

        # Ardupilot FDM communication type:
        self.model = "JSON"
        
        # Create a temporary filesystem for ardupilot to write data to/from (and modify the origin rcS files)
        self.root_fs = tempfile.TemporaryDirectory()

        # Environment for the SITL processes: a copy without Isaac Sim's Python/library paths
        self.environment = clean_sitl_environment()

    def _sitl_already_exists(self):
        return os.path.exists(f'{self.ardupilot_dir}/build/sitl/bin/{sitl_binary(self.vehicle)}')
    
    def _get_vehicle_frame(self):
        return self.ardupilot_model
    
    def launch_ardupilot(self):
        """
        Method that will launch a ardupilot instance with the specified configuration
        """
        # sim_vehicle.py -v <vehicle> -f <frame> --model JSON --console --map
        command = [
            "python3", f"{self.ardupilot_dir}/Tools/autotest/sim_vehicle.py",
            "-v", f"{self.vehicle}",
            "-f", f"{self._get_vehicle_frame()}",
            "--model", f"{self.model}",
            f"{'--no-rebuild' if self._sitl_already_exists() else ''}",
            "-I", f"{self.vehicle_id}",
            "--sysid", f"{self.vehicle_id + 1}",
            "--out", f"udp:127.0.0.1:{14550 + self.vehicle_id * 10}",
        ]
        # HIL_LEO_SILVIO_SERIAL_LINKS
        if os.environ.get("HIL_ENABLE_MAVLINK_SERIALS") == "1":
            serial_args = ["--serial2=udpclient:127.0.0.1:14561"]
            cm5_host = os.environ.get("HIL_CM5_HOST")
            if cm5_host:
                serial_args.insert(0, f"--serial1=udpclient:{cm5_host}:14560")
            command.extend(["-A", f'"{" ".join(serial_args)}"'])
            param_file = os.environ.get("HIL_SITL_PARAM_FILE")
            if param_file:
                command.extend(["--add-param-file", f'"{param_file}"'])
        # HIL_LEO_SILVIO_SITL_HOME: "lat,lon,alt,heading" (home = Isaac origin, e.g. for a
        # georeferenced terrain); unset = sim_vehicle.py default location
        home = os.environ.get("HIL_SITL_HOME")
        if home:
            command.append(f"--custom-location={home}")
        # HIL_LEO_SILVIO_SITL_CONSOLE: without terminal windows MAVProxy reads its commands from
        # <dir>/mavproxy.in (FIFO); its console and map windows stay, the HIL dashboard embeds them
        console_dir = os.environ.get("HIL_SITL_CONSOLE_DIR")
        command.extend(["--console", "--map"])
        command: str = " ".join(command)

        if console_dir:
            self._launch_without_windows(command, console_dir)
            return

        # Run in a seperate bash window
        self.ardupilot_process = subprocess.Popen(
            # ["gnome-terminal", '--disable-factory', '--', 'bash', '-c', command],
            ["gnome-terminal", '--', 'bash', '-c', command],
            cwd=self.root_fs.name,
            shell=False,
            env=self.environment,
            preexec_fn=os.setsid
        )

    def _launch_without_windows(self, command, console_dir):
        """Run sim_vehicle.py without terminal windows: MAVProxy on a FIFO and a log file,
        the SITL binary on <dir>/sitl.log."""
        os.makedirs(console_dir, exist_ok=True)
        fifo = os.path.join(console_dir, "mavproxy.in")
        if not os.path.exists(fifo):
            os.mkfifo(fifo)
        # run_in_terminal_window.sh runs "$SITL_RITW_TERMINAL <script>" instead of an xterm.
        # It starts it in the background, so SIGINT is ignored there and the SITL binary ignores
        # it too: the wrapper kills SITL itself once Isaac (this process) or the launcher
        # process group leader (killed by kill_ardupilot) is gone.
        wrapper = os.path.join(console_dir, "sitl_terminal.sh")
        with open(wrapper, "w") as file:
            file.write(SITL_WRAPPER.format(log=os.path.join(console_dir, "sitl.log"), isaac=os.getpid()))
        os.chmod(wrapper, 0o755)
        environment = dict(self.environment)
        environment["SITL_RITW_TERMINAL"] = wrapper
        environment["PYTHONUNBUFFERED"] = "1"  # MAVProxy output reaches the log at once
        # Read-write: the open does not wait for a writer and the FIFO never reaches EOF
        stdin = os.open(fifo, os.O_RDWR)
        try:
            with open(os.path.join(console_dir, "mavproxy.log"), "wb") as log:
                self.ardupilot_process = subprocess.Popen(
                    ["bash", "-c", command],
                    cwd=self.root_fs.name,
                    stdin=stdin,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    preexec_fn=os.setsid,
                )
        finally:
            os.close(stdin)

    def kill_ardupilot(self):
        """
        Method that will kill a ardupilot instance with the specified configuration
        """
        if self.ardupilot_process is not None:
            os.killpg(self.ardupilot_process.pid, signal.SIGINT)
            self.ardupilot_process.kill()
            self.ardupilot_process.wait()
            self.ardupilot_process = None

        # Define the keywords to search for in process names
        keywords = sorted(set(SITL_BINARIES.values()) | {'mavproxy'})

        # Get the list of all running processes using the ps command
        ps_output = subprocess.run(['ps', '-aux'], capture_output=True, text=True)

        # Filter processes that match any of the keywords (case insensitive)
        matching_processes = [
            line for line in ps_output.stdout.splitlines()
            if any(keyword in line.lower() for keyword in keywords)
        ]

        # Extract process IDs (PID) from the filtered lines
        pids = [line.split()[1] for line in matching_processes]

        # Kill each matching process by its PID
        for pid in pids:
            try:
                os.kill(int(pid), 9)
                print(f"Killed process {pid}")
            except ProcessLookupError:
                print(f"Process {pid} not found.")
            except PermissionError:
                print(f"Permission denied to kill process {pid}.")
            except Exception as e:
                print(f"Failed to kill process {pid}: {e}")

    def __del__(self):
        """
        If the ardupilot process is still running when the Ardupilot launch tool object is whiped from memory, then make sure
        we kill the ardupilot instance so we don't end up with hanged ardupilot instances
        """

        # Make sure the Ardupilot process gets killed
        if self.ardupilot_process:
            self.kill_ardupilot()

        # Make sure we clean the temporary filesystem used for the simulation
        self.root_fs.cleanup()


# ---- Code used for debugging the ardupilot tool ----
def main():

    ardupilot_tool = ArduPilotLaunchTool(os.environ["HOME"] + "/ardupilot")
    print("Launching ArduPilot")
    ardupilot_tool.launch_ardupilot()
    
    import time
    
    # print("Killing ArduPilot in 5")
    time.sleep(3000)
    # ardupilot_tool.kill_ardupilot()


if __name__ == "__main__":
    main()
"""Ambiente dei processi SITL: niente percorsi Python/librerie di Isaac Sim."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Caricato dal file: il pacchetto backends importa Isaac (carb) all'import.
_PATH = (
    Path(__file__).resolve().parents[2]
    / "pegasus/simulator/logic/backends/tools/ardupilot_launch_tool.py"
)
pytest.importorskip("psutil")
_spec = importlib.util.spec_from_file_location("ardupilot_launch_tool", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

ISAAC = "/opt/isaac-test"


def _env(**extra):
    return {"ISAAC_PATH": ISAAC, "HOME": "/home/u", "PATH": "/usr/bin", **extra}


def test_drops_isaac_python_and_preload():
    env = tool.clean_sitl_environment(
        _env(PYTHONHOME=f"{ISAAC}/kit/python", LD_PRELOAD=f"{ISAAC}/kit/libcarb.so")
    )
    assert "PYTHONHOME" not in env
    assert "LD_PRELOAD" not in env


def test_filters_isaac_entries_and_keeps_others():
    env = tool.clean_sitl_environment(
        _env(
            PYTHONPATH=os.pathsep.join(
                [f"{ISAAC}/kit/python/lib/python3.11", "/opt/ros/jazzy/lib/python3.12/site-packages"]
            ),
            LD_LIBRARY_PATH=os.pathsep.join([f"{ISAAC}/.", f"{ISAAC}/exts/x/lib", "/opt/ros/jazzy/lib"]),
        )
    )
    assert env["PYTHONPATH"] == "/opt/ros/jazzy/lib/python3.12/site-packages"
    assert env["LD_LIBRARY_PATH"] == "/opt/ros/jazzy/lib"


def test_removes_lists_left_empty():
    env = tool.clean_sitl_environment(
        _env(PYTHONPATH=f"{ISAAC}/kit/python/lib/python3.11", LD_LIBRARY_PATH=f"{ISAAC}/.")
    )
    assert "PYTHONPATH" not in env
    assert "LD_LIBRARY_PATH" not in env


def test_prefix_is_not_a_false_match():
    env = tool.clean_sitl_environment(_env(PYTHONPATH=f"{ISAAC}-other/lib"))
    assert env["PYTHONPATH"] == f"{ISAAC}-other/lib"


def test_running_interpreter_counts_as_isaac():
    env = tool.clean_sitl_environment(
        {"PATH": "/usr/bin", "PYTHONPATH": f"{sys.prefix}/lib/python3.11"}
    )
    assert "PYTHONPATH" not in env


def test_does_not_touch_the_source_environment():
    source = _env(PYTHONHOME=f"{ISAAC}/kit/python", PATH="/usr/bin")
    tool.clean_sitl_environment(source)
    assert source["PYTHONHOME"] == f"{ISAAC}/kit/python"
    assert tool.clean_sitl_environment(source)["PATH"] == "/usr/bin"


def test_sitl_binary_per_vehicle():
    assert tool.sitl_binary("ArduCopter") == "arducopter"
    assert tool.sitl_binary("ArduPlane") == "arduplane"
    assert tool.sitl_binary("Blimp") == "blimp"


def test_default_vehicle_is_copter_and_binary_check_follows_the_vehicle(tmp_path):
    (tmp_path / "build/sitl/bin").mkdir(parents=True)
    (tmp_path / "build/sitl/bin/arduplane").write_text("")
    copter = tool.ArduPilotLaunchTool(str(tmp_path))
    plane = tool.ArduPilotLaunchTool(str(tmp_path), vehicle="ArduPlane")
    assert copter.vehicle == "ArduCopter" and not copter._sitl_already_exists()
    assert plane._sitl_already_exists()


def _launch(monkeypatch, tmp_path, console_dir):
    calls = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))
            self.pid = 0

    monkeypatch.setattr(tool.subprocess, "Popen", FakePopen)
    monkeypatch.delenv("HIL_ENABLE_MAVLINK_SERIALS", raising=False)
    if console_dir:
        monkeypatch.setenv("HIL_SITL_CONSOLE_DIR", str(console_dir))
    else:
        monkeypatch.delenv("HIL_SITL_CONSOLE_DIR", raising=False)
    monkeypatch.setenv("DISPLAY", ":1")
    launcher = tool.ArduPilotLaunchTool(str(tmp_path / "ardupilot"))
    launcher.launch_ardupilot()
    launcher.ardupilot_process = None  # niente kill nel distruttore
    return calls[0]


def test_default_launch_opens_terminal_console_and_map(monkeypatch, tmp_path):
    args, _ = _launch(monkeypatch, tmp_path, None)
    assert args[0] == "gnome-terminal"
    assert "--console --map" in args[-1]



def test_home_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HIL_SITL_HOME", "-35.363261,149.16523,584.0,353.0")
    args, _ = _launch(monkeypatch, tmp_path, None)
    assert "--custom-location=-35.363261,149.16523,584.0,353.0" in args[-1]
    monkeypatch.delenv("HIL_SITL_HOME")
    args, _ = _launch(monkeypatch, tmp_path, None)
    assert "--custom-location" not in args[-1]

def test_console_dir_launch_has_no_windows(monkeypatch, tmp_path):
    console = tmp_path / "console"
    args, kwargs = _launch(monkeypatch, tmp_path, console)
    assert args[:2] == ["bash", "-c"]
    assert "--console --map" in args[2]  # finestre di MAVProxy, inglobate dalla dashboard
    assert kwargs["env"]["SITL_RITW_TERMINAL"] == str(console / "sitl_terminal.sh")
    assert os.access(console / "sitl_terminal.sh", os.X_OK)  # niente xterm per SITL
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert (console / "mavproxy.log").exists()
    assert os.path.exists(console / "mavproxy.in")


def test_wrapper_kills_sitl_when_isaac_exits(tmp_path):
    import subprocess
    import time

    marker = "47.31"  # durata unica: identifica il finto SITL tra i processi
    # Come lo script di run_in_terminal_window.sh: SITL in primo piano
    script = tmp_path / "ritw.sh"
    script.write_text(f"#!/bin/sh\nsleep {marker}\n")
    script.chmod(0o755)
    isaac = subprocess.Popen(["sleep", "30"])  # finto Isaac
    wrapper = tmp_path / "sitl_terminal.sh"
    wrapper.write_text(tool.SITL_WRAPPER.format(log=tmp_path / "sitl.log", isaac=isaac.pid))
    wrapper.chmod(0o755)
    runner = subprocess.Popen([str(wrapper), str(script)])

    def sitl_pids():
        out = subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True)
        return out.stdout.split()

    deadline = time.monotonic() + 5
    while not sitl_pids() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sitl_pids(), "SITL finto non partito"
    assert runner.poll() is None  # Isaac vivo: SITL resta
    isaac.kill()
    isaac.wait()
    runner.wait(timeout=5)  # il wrapper esce entro ~1 s dalla fine di Isaac
    deadline = time.monotonic() + 2
    while sitl_pids() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not sitl_pids()

"""
| File: multirotor.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the Multirotor class which is used as the base for all the multirotor vehicles.
"""

import numpy as np

import carb
from omni.isaac.dynamic_control import _dynamic_control

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle import Vehicle

# Mavlink interface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS

class MultirotorConfig:
    """
    A data class that is used for configuring a Multirotor
    """

    def __init__(self):
        """
        Initialization of the MultirotorConfig class
        """

        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "quadrotor"

        # The USD file that describes the visual aspect of the vehicle (and some properties such as mass and moments of inertia)
        self.usd_file = ""

        # The default thrust curve for a quadrotor and dynamics relating to drag
        self.thrust_curve = QuadraticThrustCurve()
        self.drag = LinearDrag([0.50, 0.30, 0.0])

        # The default sensors for a quadrotor
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]

        # The default graphical sensors for a quadrotor
        self.graphical_sensors = []

        # The default omnigraphs for a quadrotor
        self.graphs = []

        # The backends for actually sending commands to the vehicle. By default use mavlink (with default mavlink configurations)
        # [Can be None as well, if we do not desired to use PX4 with this simulated vehicle]. It can also be a ROS2 backend
        # or your own custom Backend implementation!
        self.backends = [PX4MavlinkBackend(config=PX4MavlinkBackendConfig())]

        # Optional advanced physics (pegasus.simulator.logic.physics, requires fastuav).
        # None: quadratic thrust curve + linear drag above. A PhysicsConfig or a dict (even
        # partial): motor, propeller (Ct/Cp with advance ratio and incidence), battery and
        # airframe aerodynamics, with mass, CG and inertia written to the USD at spawn.
        self.physics = None


class Multirotor(Vehicle):
    """Multirotor class - It defines a base interface for creating a multirotor
    """
    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "quadrotor",
        usd_file: str = "",
        vehicle_id: int = 0,
        # Spawning pose of the vehicle
        init_pos=[0.0, 0.0, 0.07],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        config=MultirotorConfig(),
    ):
        """Initializes the multirotor object

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_file (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            vehicle_id (int): The id to be used for the vehicle. Defaults to 0.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (MultirotorConfig, optional): Defaults to MultirotorConfig().
        """

        # 1. Initiate the Vehicle object itself
        super().__init__(stage_prefix, usd_file, init_pos, init_orientation, config.sensors, config.graphical_sensors, config.graphs, config.backends)

        # 2. Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._thrusters = config.thrust_curve
        self._drag = config.drag

        # Optional advanced physics (see MultirotorConfig.physics)
        self._physics = None
        physics = getattr(config, "physics", None)
        if physics is not None:
            self._setup_advanced_physics(physics)

    def start(self):
        """Resets the advanced physics state (rotor speeds, battery) when the simulation starts"""
        if self._physics is not None:
            self._physics.reset(0.0)
            self._physics_time_s = 0.0

    def stop(self):
        """Closes the advanced physics log when the simulation stops"""
        if self._physics is not None:
            self._physics_logger.close()

    def update(self, dt: float):
        """
        Method that computes and applies the forces to the vehicle in simulation based on the motor speed. 
        This method must be implemented by a class that inherits this type. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Get the articulation root of the vehicle
        if self._physics is not None:
            self._update_advanced_physics(dt)
            return

        articulation = self.get_dc_interface().get_articulation(self._stage_prefix)

        # Get the desired angular velocities for each rotor from the first backend (can be mavlink or other) expressed in rad/s
        if len(self._backends) != 0:
            desired_rotor_velocities = self._backends[0].input_reference()
        else:
            desired_rotor_velocities = [0.0 for i in range(self._thrusters._num_rotors)]

        # Input the desired rotor velocities in the thruster model
        self._thrusters.set_input_reference(desired_rotor_velocities)

        # Get the desired forces to apply to the vehicle
        forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)

        # Apply force to each rotor
        for i in range(4):

            # Apply the force in Z on the rotor frame
            self.apply_force([0.0, 0.0, forces_z[i]], body_part="/rotor" + str(i))

            # Generate the rotating propeller visual effect
            self.handle_propeller_visual(i, forces_z[i], articulation)

        # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
        self.apply_torque([0.0, 0.0, rolling_moment], "/body")

        # Compute the total linear drag force to apply to the vehicle's body frame
        drag = self._drag.update(self._state, dt)
        self.apply_force(drag, body_part="/body")

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)

    # --------------------------------------------------------------------
    # -------------------- Advanced physics (optional) -------------------
    # --------------------------------------------------------------------

    def _setup_advanced_physics(self, physics):
        """Builds the advanced physics model and writes mass properties to the USD.

        Args:
            physics (PhysicsConfig | dict): see pegasus.simulator.logic.physics.config
        """
        from pegasus.simulator.logic.physics.config import PhysicsConfig
        from pegasus.simulator.logic.physics.log import PhysicsLogger
        from pegasus.simulator.logic.physics.model import AdvancedMultirotorPhysics

        config = PhysicsConfig.coerce(physics)
        self._physics = AdvancedMultirotorPhysics(config)
        self._physics_logger = PhysicsLogger(config.log)
        self._physics_time_s = 0.0
        self.last_wrench = None
        # Equivalent quadratic curve: only for the rotor directions (propeller animation) and
        # force_and_torques_to_velocities. Forces come from the advanced model.
        self._thrusters = QuadraticThrustCurve(self._physics.equivalent_thrust_curve())
        self._drag = None
        self._apply_mass_properties(config)

    def _apply_mass_properties(self, config):
        """Writes mass, CG and inertia of the complete drone (config.rigid_body) to the USD.

        The rotors keep their USD mass; the `body` prim gets the rest.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        from pegasus.simulator.logic.physics.provenance import fastuav_version
        from pegasus.simulator.logic.physics.rigid_body import body_mass_properties, rotor_layout_mismatch

        stage = self._current_stage
        body = stage.GetPrimAtPath(self._stage_prefix + "/body")
        if not body.IsValid():
            raise ValueError(f"prim {self._stage_prefix}/body not found: the USD is not a Pegasus multirotor")
        to_body = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0).GetInverse()
        rotors = []
        for i in range(4):
            prim = stage.GetPrimAtPath(f"{self._stage_prefix}/rotor{i}")
            if not prim.IsValid():
                raise ValueError(f"prim {self._stage_prefix}/rotor{i} not found")
            world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            position = np.array(to_body.Transform(world.ExtractTranslation()))
            mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr() if prim.HasAPI(UsdPhysics.MassAPI) else None
            mass = float(mass_attr.Get()) if mass_attr and mass_attr.Get() is not None else 0.0
            rotors.append((mass, position, None))

        props = body_mass_properties(config.rigid_body, rotors)
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr().Set(float(props.mass_kg))
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in props.center_of_mass_m]))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in props.diagonal_inertia]))
        x, y, z, w = (float(v) for v in props.principal_axes_xyzw)
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(w, x, y, z))
        mass_api.CreateDensityAttr().Set(0.0)

        self.mass_properties = {"body": props, "rotor_masses_kg": [m for m, _, _ in rotors],
                                "rotor_positions_m": [p for _, p, _ in rotors]}
        carb.log_info(
            f"[Multirotor] advanced physics: total mass {config.rigid_body.mass_kg:.3f} kg "
            f"(body {props.mass_kg:.3f} + rotors {[round(m, 4) for m, _, _ in rotors]}), "
            f"body CG {[round(float(v), 4) for v in props.center_of_mass_m]}, "
            f"principal inertia {[round(float(v), 5) for v in props.diagonal_inertia]}")
        for problem in rotor_layout_mismatch(config.rotors.positions_m, self.mass_properties["rotor_positions_m"]):
            carb.log_warn(f"[Multirotor] physics.rotors.positions_m differs from the USD: {problem}")
        try:
            carb.log_info(f"[Multirotor] fastuav: {fastuav_version()}")
        except Exception as error:  # informative only
            carb.log_warn(f"[Multirotor] fastuav version not available: {error}")

    def _update_advanced_physics(self, dt: float):
        """Physics step with the advanced model: rotor thrusts, aerodynamic force and torques.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        from pegasus.simulator.logic.physics.frames import BodyState

        articulation = self.get_dc_interface().get_articulation(self._stage_prefix)
        if len(self._backends) != 0:
            commands = self._backends[0].input_reference()
        else:
            commands = [0.0] * 4

        state = BodyState.from_pegasus(self._state)
        wrench = self._physics.step(state, commands, dt)
        self.last_wrench = wrench

        # Thrust of each rotor along its +z axis
        for i in range(4):
            thrust = float(wrench.rotor_thrust_n[i])
            self.apply_force([0.0, 0.0, thrust], body_part="/rotor" + str(i))
            self.handle_propeller_visual(i, thrust, articulation)

        # Aerodynamic force at the CG, rotor reaction + aerodynamic moment on the body
        if np.any(wrench.body_force_n):
            self.apply_force([float(v) for v in wrench.body_force_n],
                             pos=[float(v) for v in self._physics.cg], body_part="/body")
        self.apply_torque([float(v) for v in wrench.body_torque_nm], "/body")

        self._physics_time_s += dt
        self._physics_logger.write(self._physics_time_s, state, wrench)

        for backend in self._backends:
            backend.update(dt)

    def handle_propeller_visual(self, rotor_number, force: float, articulation):
        """
        Auxiliar method used to set the joint velocity of each rotor (for animation purposes) based on the 
        amount of force being applied on each joint

        Args:
            rotor_number (int): The number of the rotor to generate the rotation animation
            force (float): The force that is being applied on that rotor
            articulation (_type_): The articulation group the joints of the rotors belong to
        """

        # Rotate the joint to yield the visual of a rotor spinning (for animation purposes only)
        joint = self.get_dc_interface().find_articulation_dof(articulation, "joint" + str(rotor_number))

        # Spinning when armed but not applying force
        if 0.0 < force < 0.1:
            self.get_dc_interface().set_dof_velocity(joint, 5 * self._thrusters.rot_dir[rotor_number])
        # Spinning when armed and applying force
        elif 0.1 <= force:
            self.get_dc_interface().set_dof_velocity(joint, 100 * self._thrusters.rot_dir[rotor_number])
        # Not spinning
        else:
            self.get_dc_interface().set_dof_velocity(joint, 0)

    def force_and_torques_to_velocities(self, force: float, torque: np.ndarray):
        """
        Auxiliar method used to get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] to be applied in the multirotor's body frame.

        Note: This method assumes a quadratic thrust curve. This method will be improved in a future update,
        and a general thrust allocation scheme will be adopted. For now, it is made to work with multirotors directly.

        Args:
            force (np.ndarray): A vector of the force to be applied in the body frame of the vehicle [N]
            torque (np.ndarray): A vector of the torque to be applied in the body frame of the vehicle [Nm]

        Returns:
            list: A list of angular velocities [rad/s] to apply in reach rotor to accomplish suchs forces and torques
        """

        # Get the body frame of the vehicle
        rb = self.get_dc_interface().get_rigid_body(self._stage_prefix + "/body")

        # Get the rotors of the vehicle
        rotors = [self.get_dc_interface().get_rigid_body(self._stage_prefix + "/rotor" + str(i)) for i in range(self._thrusters._num_rotors)]

        # Get the relative position of the rotors with respect to the body frame of the vehicle (ignoring the orientation for now)
        relative_poses = self.get_dc_interface().get_relative_body_poses(rb, rotors)

        # Define the alocation matrix
        aloc_matrix = np.zeros((4, self._thrusters._num_rotors))
        
        # Define the first line of the matrix (T [N])
        aloc_matrix[0, :] = np.array(self._thrusters._rotor_constant)                                           

        # Define the second and third lines of the matrix (\tau_x [Nm] and \tau_y [Nm])
        aloc_matrix[1, :] = np.array([relative_poses[i].p[1] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])
        aloc_matrix[2, :] = np.array([-relative_poses[i].p[0] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])

        # Define the forth line of the matrix (\tau_z [Nm])
        aloc_matrix[3, :] = np.array([self._thrusters._rolling_moment_coefficient[i] * self._thrusters._rot_dir[i] for i in range(self._thrusters._num_rotors)])

        # Compute the inverse allocation matrix, so that we can get the angular velocities (squared) from the total thrust and torques
        aloc_inv = np.linalg.pinv(aloc_matrix)

        # Compute the target angular velocities (squared)
        squared_ang_vel = aloc_inv @ np.array([force, torque[0], torque[1], torque[2]])

        # Making sure that there is no negative value on the target squared angular velocities
        squared_ang_vel[squared_ang_vel < 0] = 0.0

        # ------------------------------------------------------------------------------------------------
        # Saturate the inputs while preserving their relation to each other, by performing a normalization
        # ------------------------------------------------------------------------------------------------
        max_thrust_vel_squared = np.power(self._thrusters.max_rotor_velocity[0], 2)
        max_val = np.max(squared_ang_vel)

        if max_val >= max_thrust_vel_squared:
            normalize = np.maximum(max_val / max_thrust_vel_squared, 1.0)

            squared_ang_vel = squared_ang_vel / normalize

        # Compute the angular velocities for each rotor in [rad/s]
        ang_vel = np.sqrt(squared_ang_vel)

        return ang_vel

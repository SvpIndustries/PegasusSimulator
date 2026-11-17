"""Il drone montato: 4 gruppi motore+elica, batteria, airframe -> forze e coppie a ogni passo.

Non integra il moto: lo fa PhysX (in Isaac, tramite Multirotor) o `sim6dof` (offline).
Riceve lo stato del corpo e i comandi di giri, aggiorna il proprio stato interno (giri
reali, scarica della batteria) e restituisce le forze da applicare.

Convenzione della coppia di reazione (la stessa di Pegasus, `rot_dir`): un rotore "cw"
visto da +z spinge il corpo con +Q attorno a z, un "ccw" con -Q.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, pi
from typing import Any

import numpy as np

from pegasus.simulator.logic.physics.aero import AeroLoad, Airframe
from pegasus.simulator.logic.physics.battery import Battery
from pegasus.simulator.logic.physics.frames import E3, BodyState, air_data
from pegasus.simulator.logic.physics.motor import Motor
from pegasus.simulator.logic.physics.propeller import Propeller, RotorLoad
from pegasus.simulator.logic.physics.config import PhysicsConfig


@dataclass
class Wrench:
    """Cosa applicare al passo: spinte sui rotori e forza/coppia sul corpo (assi corpo)."""

    rotor_thrust_n: np.ndarray  # lungo +z di ogni rotore
    rotor_omega_rad_s: np.ndarray
    body_force_n: np.ndarray  # aerodinamica, applicata al baricentro
    body_torque_nm: np.ndarray  # reazione dei rotori + momento aerodinamico
    telemetry: dict[str, float] = field(default_factory=dict)


class AdvancedMultirotorPhysics:
    def __init__(self, physics: PhysicsConfig) -> None:
        self.physics = physics
        self.rigid_body = rigid_body = physics.rigid_body
        self.cg = np.asarray(rigid_body.cg_m, dtype=float)
        self.positions = np.array(physics.rotors.positions_m, dtype=float)
        self.arms = self.positions - self.cg
        self.directions = np.array([1.0 if d == "cw" else -1.0 for d in physics.rotors.directions])
        self.propeller = Propeller(physics.propeller, physics.atmosphere)
        self.motors = [Motor(physics.motor, physics.esc) for _ in range(4)]
        self.battery = Battery(physics.battery)
        self.airframe = Airframe(physics.aero, physics.atmosphere, float(self.cg[2]))
        self.wind = np.asarray(physics.atmosphere.wind_m_s, dtype=float)
        self._loads: list[RotorLoad | None] = [None] * 4

    def reset(self, omega: float = 0.0) -> None:
        for motor in self.motors:
            motor.reset(omega)
        self.battery.reset()
        self._loads = [None] * 4

    @property
    def mass_kg(self) -> float:
        return self.rigid_body.mass_kg

    def hover_omega(self) -> float:
        """Giri per sostenere il peso con 4 rotori a punto fisso (stima per il controllore)."""
        k_t, _ = self.propeller.static_constants()
        return float(np.sqrt(self.mass_kg * 9.80665 / (4.0 * k_t)))

    def step(self, state: BodyState, omega_cmd: list[float] | np.ndarray, dt: float) -> Wrench:
        rotation = state.rotation
        air = air_data(state.linear_velocity, rotation, self.wind)
        w_body = np.asarray(state.angular_velocity, dtype=float)
        bus_v = self.battery.voltage_v
        bus_i = self.battery.current_a

        thrust = np.zeros(4)
        omega = np.zeros(4)
        torque = np.zeros(3)
        bus_power = 0.0
        tele: dict[str, float] = {}
        for i, motor in enumerate(self.motors):
            u_rotor = air.u + np.cross(w_body, self.arms[i])
            v_axial, v_edge = float(u_rotor[2]), hypot(u_rotor[0], u_rotor[1])
            omega_max = min(motor.speed_limit(bus_v, bus_i / 4.0),
                            self.propeller.max_speed(v_axial, v_edge))
            previous = self._loads[i]
            motor.step(float(omega_cmd[i]), omega_max, previous.torque_nm if previous else 0.0, dt)
            load = self.propeller.load(motor.omega, v_axial, v_edge)
            self._loads[i] = load
            electrics = motor.electrics(load.torque_nm, motor.omega, bus_v)
            thrust[i] = load.thrust_n
            omega[i] = motor.omega
            torque[2] += self.directions[i] * load.torque_nm
            bus_power += electrics.bus_power_w
            tele[f"omega{i}"] = motor.omega
            tele[f"thrust{i}"] = load.thrust_n
            tele[f"J{i}"] = load.advance_ratio
            tele[f"current{i}"] = electrics.current_a
        self.battery.step(bus_power, dt)
        aero: AeroLoad = self.airframe.load(air)
        torque += aero.torque_nm
        tele.update({
            "airspeed": air.airspeed, "body_incidence": air.body_incidence,
            "lift": aero.lift_n, "drag": aero.drag_n, "pitching_moment": aero.pitching_moment_nm,
            "bus_voltage": self.battery.voltage_v, "bus_current": self.battery.current_a,
            "bus_power": bus_power, "dod": self.battery.depth_of_discharge,
        })
        return Wrench(thrust, omega, aero.force_n, torque, tele)

    def total_about_cg(self, wrench: Wrench) -> tuple[np.ndarray, np.ndarray]:
        """Forza e coppia totali attorno al baricentro, assi corpo (per l'integratore offline)."""
        force = wrench.body_force_n + wrench.rotor_thrust_n.sum() * E3
        torque = wrench.body_torque_nm.copy()
        for arm, t in zip(self.arms, wrench.rotor_thrust_n, strict=True):
            torque += np.cross(arm, t * E3)
        return force, torque

    def equivalent_thrust_curve(self) -> dict[str, Any]:
        """Config di una QuadraticThrustCurve equivalente a punto fisso.

        Non calcola le forze (le calcola `step`): serve al Multirotor per i versi dei rotori
        (animazione delle eliche) e a chi usa `force_and_torques_to_velocities`.
        """
        k_t, k_q = self.propeller.static_constants()
        battery = self.physics.battery
        omega_max = self.physics.motor.kv_rpm_per_v * 2.0 * pi / 60.0 * (
            battery.cells_series * battery.v_full_v)
        return {
            "num_rotors": 4,
            "rotor_constant": [k_t] * 4,
            "rolling_moment_coefficient": [k_q] * 4,
            "rot_dir": [int(d) for d in self.directions],
            "min_rotor_velocity": [0.0] * 4,
            "max_rotor_velocity": [omega_max] * 4,
        }

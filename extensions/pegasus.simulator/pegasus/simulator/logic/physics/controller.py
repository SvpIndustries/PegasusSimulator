"""Controllore interno di prova: geometrico su SO(3) (Lee, Leok, McClamroch 2010).

Serve a far volare il drone mentre si prova la fisica, senza ArduPilot. Esce in giri
desiderati [rad/s], come un backend di Pegasus.

Modalità:
- hover:      tiene `setpoint_m` e `yaw_rad`;
- waypoint:   insegue `waypoints_m` a velocità limitata, poi tiene l'ultimo;
- transition: tailsitter. Fino a `transition_start_s` tiene il setpoint; poi il muso
              (+z corpo) si inclina verso +x mondo fino a `transition_pitch_rad` in
              `transition_duration_s`, mentre la spinta tiene la quota. L'assetto è
              imposto, la posizione orizzontale è libera (il drone accelera in avanti).

L'allocazione usa le costanti a punto fisso dell'elica (k_T, k_Q): è un controllore,
non deve conoscere la fisica fine (J, incidenza, batteria), che resta nel modello.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import cos, sin
from typing import Any

import numpy as np

from pegasus.simulator.logic.physics.config import Vec3, from_dict
from pegasus.simulator.logic.physics.frames import E3, BodyState, rot_y, rot_z, vee
from pegasus.simulator.logic.physics.model import AdvancedMultirotorPhysics
from pegasus.simulator.logic.physics.rigid_body import inertia_matrix

G = 9.80665


@dataclass
class GeometricControllerConfig:
    """Parametri del controllore di prova (GeometricControllerBackend)."""

    mode: str = "hover"  # "hover" | "waypoint" | "transition"
    setpoint_m: Vec3 = (0.0, 0.0, 2.0)  # punto da tenere [x, y, z] ENU
    yaw_rad: float = 0.0
    waypoints_m: list[Vec3] = field(default_factory=list)  # punti in modalità waypoint
    waypoint_radius_m: float = 0.5  # raggio di arrivo a un waypoint
    max_speed_m_s: float = 3.0  # velocità massima in waypoint
    # transizione del tailsitter: il muso (+z corpo) si inclina verso +x mondo
    transition_start_s: float = 3.0
    transition_duration_s: float = 4.0  # durata della rampa di assetto
    transition_pitch_rad: float = 1.2  # inclinazione finale
    kp_position: Vec3 = (4.0, 4.0, 6.0)
    kd_velocity: Vec3 = (3.0, 3.0, 4.0)
    # anello d'assetto ~5 volte più veloce di quello di posizione (w_n ~11 rad/s, zeta ~0,9)
    k_attitude: Vec3 = (120.0, 120.0, 40.0)  # guadagni di assetto (×inerzia)
    k_rate: Vec3 = (20.0, 20.0, 12.0)  # guadagni di velocità angolare (×inerzia)
    max_tilt_rad: float = 0.6  # inclinazione massima in hover/waypoint

    def __post_init__(self) -> None:
        if self.mode not in ("hover", "waypoint", "transition"):
            raise ValueError(f"mode deve essere hover, waypoint o transition, non {self.mode!r}")

    @classmethod
    def coerce(cls, value: GeometricControllerConfig | Mapping[str, Any] | None
               ) -> GeometricControllerConfig:
        return value if isinstance(value, cls) else from_dict(cls, value)


def allocation_matrix(arms: np.ndarray, directions: np.ndarray, k_t: float,
                      k_q: float) -> np.ndarray:
    """[T, Mx, My, Mz] = A @ w² per rotori con spinta lungo +z ai bracci `arms` (dal baricentro)."""
    a = np.zeros((4, len(arms)))
    for i, (r, d) in enumerate(zip(arms, directions, strict=True)):
        a[0, i] = k_t
        a[1:, i] = np.cross(r, E3) * k_t
        a[3, i] += d * k_q
    return a


class GeometricController:
    def __init__(self, settings: GeometricControllerConfig, mass_kg: float, inertia: np.ndarray,
                 arms: np.ndarray, directions: np.ndarray, k_t: float, k_q: float,
                 omega_max: float) -> None:
        self.s = settings
        self.mass = mass_kg
        self.inertia = inertia
        self.alloc_inv = np.linalg.pinv(allocation_matrix(arms, directions, k_t, k_q))
        self.omega_max = omega_max
        self.waypoint_index = 0
        self.time_s = 0.0

    def reset(self) -> None:
        self.waypoint_index = 0
        self.time_s = 0.0

    # ---------------------------------------------------------------- riferimenti

    def _position_reference(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(posizione desiderata, velocità desiderata)."""
        s = self.s
        if s.mode != "waypoint" or not s.waypoints_m:
            return np.asarray(s.setpoint_m, dtype=float), np.zeros(3)
        target = np.asarray(s.waypoints_m[self.waypoint_index], dtype=float)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance < s.waypoint_radius_m and self.waypoint_index < len(s.waypoints_m) - 1:
            self.waypoint_index += 1
        if distance > 1e-6 and self.waypoint_index < len(s.waypoints_m) - 1:
            return target, delta / distance * min(s.max_speed_m_s, distance)
        return target, np.zeros(3)

    def transition_pitch(self) -> float:
        s = self.s
        if s.mode != "transition" or self.time_s < s.transition_start_s:
            return 0.0
        fraction = min((self.time_s - s.transition_start_s) / s.transition_duration_s, 1.0)
        return fraction * s.transition_pitch_rad

    # ---------------------------------------------------------------- legge di controllo

    def update(self, state: BodyState, dt: float) -> np.ndarray:
        """Giri desiderati dei 4 rotori [rad/s]."""
        self.time_s += dt
        s = self.s
        r = state.rotation
        p = np.asarray(state.position, dtype=float)
        v = np.asarray(state.linear_velocity, dtype=float)
        w = np.asarray(state.angular_velocity, dtype=float)
        p_d, v_d = self._position_reference(p)
        e_p, e_v = p - p_d, v - v_d
        kp, kd = np.asarray(s.kp_position), np.asarray(s.kd_velocity)
        pitch = self.transition_pitch()

        if pitch > 0.0:
            # tailsitter: assetto imposto, la spinta tiene la quota
            r_d = rot_z(s.yaw_rad) @ rot_y(pitch)
            a_z = -kp[2] * e_p[2] - kd[2] * e_v[2] + G
            thrust = self.mass * a_z / max(float((r @ E3)[2]), 0.25)
        else:
            f_des = self.mass * (-kp * e_p - kd * e_v + G * E3)
            # limite d'inclinazione: si taglia la parte orizzontale
            horizontal = np.hypot(f_des[0], f_des[1])
            max_h = max(f_des[2], 0.1) * np.tan(s.max_tilt_rad)
            if horizontal > max_h:
                f_des[:2] *= max_h / horizontal
            b3 = f_des / np.linalg.norm(f_des)
            b1c = np.array([cos(s.yaw_rad), sin(s.yaw_rad), 0.0])
            b2 = np.cross(b3, b1c)
            b2 /= np.linalg.norm(b2)
            r_d = np.column_stack((np.cross(b2, b3), b2, b3))
            thrust = float(f_des @ (r @ E3))

        e_r = 0.5 * vee(r_d.T @ r - r.T @ r_d)
        j_diag = np.diag(self.inertia)
        moment = (-np.asarray(s.k_attitude) * j_diag * e_r - np.asarray(s.k_rate) * j_diag * w
                  + np.cross(w, self.inertia @ w))
        omega_sq = self.alloc_inv @ np.array([max(thrust, 0.0), *moment])
        omega_sq = np.clip(omega_sq, 0.0, self.omega_max**2)
        return np.sqrt(omega_sq)


def build_controller(config: GeometricControllerConfig,
                     model: AdvancedMultirotorPhysics) -> GeometricController:
    """Controllore tarato sul modello: massa, inerzia, bracci, costanti statiche dell'elica."""
    k_t, k_q = model.propeller.static_constants()
    battery_v = model.battery.open_circuit_voltage()
    omega_max = min(model.motors[0].speed_limit(battery_v, 0.0),
                    model.propeller.max_speed(0.0, 0.0))
    return GeometricController(
        config, model.mass_kg, inertia_matrix(model.rigid_body.inertia_kg_m2),
        model.arms, model.directions, k_t, k_q, omega_max,
    )

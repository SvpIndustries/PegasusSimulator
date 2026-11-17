"""Convenzioni degli assi e trasformazioni (numpy puro).

- Mondo: ENU, come Pegasus/Isaac.
- Corpo: FLU, come l'URDF e Pegasus. L'asse +z del corpo è l'asse di spinta dei rotori
  e, per un tailsitter, anche l'asse della fusoliera con il muso verso +z.
- Quaternioni in ordine [x, y, z, w] (Pegasus `state.attitude`), corpo -> mondo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, hypot, pi

import numpy as np

E3 = np.array([0.0, 0.0, 1.0])


def quat_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Matrice di rotazione corpo -> mondo da un quaternione [x, y, z, w]."""
    x, y, z, w = np.asarray(q_xyzw, dtype=float) / np.linalg.norm(q_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Quaternione [x, y, z, w] (w >= 0) da una matrice di rotazione."""
    m = np.asarray(r, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class BodyState:
    """Stato del corpo rigido come lo dà Pegasus (`vehicle.state`)."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m, ENU
    attitude: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s, ENU
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s, corpo

    @classmethod
    def from_pegasus(cls, state: object) -> BodyState:
        return cls(
            position=np.asarray(state.position, dtype=float),
            attitude=np.asarray(state.attitude, dtype=float),
            linear_velocity=np.asarray(state.linear_velocity, dtype=float),
            angular_velocity=np.asarray(state.angular_velocity, dtype=float),
        )

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.attitude)


@dataclass(frozen=True)
class AirData:
    """Vento relativo visto dal corpo, scomposto come lo vuole MultirotorAero.

    `u` è la velocità del CORPO rispetto all'aria, in assi corpo. Il flusso arriva da
    -u. `body_incidence` (aJ di FAST-UAV) è l'angolo fra l'asse +z e u: 0 = muso nel
    vento (crociera del tailsitter), pi/2 = di traverso (hover con vento laterale),
    pi = coda nel vento (discesa in hover). `disk_alpha` = pi/2 - aJ è l'incidenza del
    disco dei rotori nella convenzione di FAST-UAV (pi/2 = flusso assiale).
    """

    u: np.ndarray
    airspeed: float
    body_incidence: float
    normal_dir: np.ndarray  # versore della componente di u normale all'asse (0 se assiale)

    @property
    def disk_alpha(self) -> float:
        return pi / 2.0 - self.body_incidence


def air_data(velocity_world: np.ndarray, rotation: np.ndarray, wind_world: np.ndarray) -> AirData:
    """Vento relativo dal moto del corpo e dal vento (entrambi in ENU)."""
    u = rotation.T @ (np.asarray(velocity_world, dtype=float) - np.asarray(wind_world, dtype=float))
    normal = hypot(u[0], u[1])
    airspeed = float(np.linalg.norm(u))
    normal_dir = np.array([u[0] / normal, u[1] / normal, 0.0]) if normal > 1e-9 else np.zeros(3)
    return AirData(u=u, airspeed=airspeed, body_incidence=atan2(normal, u[2]),
                   normal_dir=normal_dir)

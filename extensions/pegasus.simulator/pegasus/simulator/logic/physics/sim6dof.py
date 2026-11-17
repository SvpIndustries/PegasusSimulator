"""Integratore del corpo rigido per il gemello offline (al posto di PhysX).

Stesso passo e stessa sequenza della physics callback in Isaac: legge lo stato, chiede
le forze a `AdvancedMultirotorPhysics`, integra con Eulero semi-implicito (come PhysX). Il suolo è un
piano a z = 0 che ferma la discesa: basta per le prove, non è un modello di contatto.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pegasus.simulator.logic.physics.frames import BodyState, matrix_to_quat, quat_to_matrix, skew
from pegasus.simulator.logic.physics.model import AdvancedMultirotorPhysics, Wrench
from pegasus.simulator.logic.physics.rigid_body import inertia_matrix

G = np.array([0.0, 0.0, -9.80665])


@dataclass
class Sample:
    t: float
    state: BodyState
    wrench: Wrench


@dataclass
class Trace:
    samples: list[Sample] = field(default_factory=list)

    def column(self, name: str) -> np.ndarray:
        return np.array([s.wrench.telemetry[name] for s in self.samples])

    @property
    def time(self) -> np.ndarray:
        return np.array([s.t for s in self.samples])

    @property
    def positions(self) -> np.ndarray:
        return np.array([s.state.position for s in self.samples])

    @property
    def velocities(self) -> np.ndarray:
        return np.array([s.state.linear_velocity for s in self.samples])


Command = Callable[[BodyState, float], np.ndarray]


def simulate(model: AdvancedMultirotorPhysics, command: Command, initial: BodyState, duration_s: float,
             dt: float, sample_every: int = 1) -> Trace:
    """Simula `duration_s` secondi; `command(stato, dt)` restituisce i giri desiderati."""
    mass = model.mass_kg
    inertia = inertia_matrix(model.rigid_body.inertia_kg_m2)
    inertia_inv = np.linalg.inv(inertia)
    p = np.array(initial.position, dtype=float)
    v = np.array(initial.linear_velocity, dtype=float)
    r = quat_to_matrix(initial.attitude)
    w = np.array(initial.angular_velocity, dtype=float)
    trace = Trace()
    steps = int(round(duration_s / dt))
    for k in range(steps):
        state = BodyState(p.copy(), matrix_to_quat(r), v.copy(), w.copy())
        wrench = model.step(state, command(state, dt), dt)
        force_b, torque_b = model.total_about_cg(wrench)
        v = v + (r @ force_b / mass + G) * dt
        w = w + inertia_inv @ (torque_b - np.cross(w, inertia @ w)) * dt
        p = p + v * dt
        r = r @ _expm(skew(w * dt))
        if p[2] < 0.0:  # suolo
            p[2], v[2] = 0.0, max(v[2], 0.0)
        if k % sample_every == 0:
            trace.samples.append(Sample((k + 1) * dt, state, wrench))
    return trace


def _expm(m: np.ndarray) -> np.ndarray:
    """exp di una matrice antisimmetrica (Rodrigues)."""
    theta = np.linalg.norm(vee_safe(m))
    if theta < 1e-12:
        return np.eye(3) + m
    return np.eye(3) + np.sin(theta) / theta * m + (1 - np.cos(theta)) / theta**2 * (m @ m)


def vee_safe(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])

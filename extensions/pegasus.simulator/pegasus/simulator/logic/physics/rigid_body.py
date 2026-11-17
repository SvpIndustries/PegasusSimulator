"""Massa e inerzia per il USD: dal drone completo al solo prim `body`.

La config descrive il drone COMPLETO (massa totale, baricentro, inerzia attorno al
baricentro). Nel USD i rotori sono corpi rigidi a parte con la loro massa: PhysX somma
tutto. Al prim `body` va quindi il resto, ricavato togliendo i rotori come masse puntiformi
(più la loro inerzia propria, se nota). USD vuole l'inerzia come diagonale + assi principali.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pegasus.simulator.logic.physics.frames import matrix_to_quat
from pegasus.simulator.logic.physics.config import InertiaConfig, RigidBodyConfig


def inertia_matrix(i: InertiaConfig) -> np.ndarray:
    return np.array([[i.ixx, i.ixy, i.ixz], [i.ixy, i.iyy, i.iyz], [i.ixz, i.iyz, i.izz]])


def point_mass_inertia(mass: float, offset: np.ndarray) -> np.ndarray:
    r = np.asarray(offset, dtype=float)
    return mass * (float(r @ r) * np.eye(3) - np.outer(r, r))


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    center_of_mass_m: np.ndarray
    diagonal_inertia: np.ndarray  # momenti principali
    principal_axes_xyzw: np.ndarray  # rotazione corpo -> assi principali, quaternione

    @property
    def inertia(self) -> np.ndarray:
        from pegasus.simulator.logic.physics.frames import quat_to_matrix

        r = quat_to_matrix(self.principal_axes_xyzw)
        return r @ np.diag(self.diagonal_inertia) @ r.T


def principal(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(momenti principali, quaternione degli assi) di un tensore simmetrico."""
    values, vectors = np.linalg.eigh(inertia)
    if np.linalg.det(vectors) < 0:
        vectors[:, 2] = -vectors[:, 2]  # base destrorsa
    return values, matrix_to_quat(vectors)


RotorMass = tuple[float, np.ndarray, np.ndarray | None]


def body_mass_properties(rigid_body: RigidBodyConfig,
                         rotors: list[RotorMass]) -> MassProperties:
    """Massa del solo `body` dato il drone completo e i rotori del USD.

    rotors: per ogni rotore (massa, posizione del suo baricentro nel frame del corpo,
    inerzia propria attorno al suo baricentro o None).
    """
    cg = np.asarray(rigid_body.cg_m, dtype=float)
    total = inertia_matrix(rigid_body.inertia_kg_m2)
    rotor_mass = sum(m for m, _, _ in rotors)
    body_mass = rigid_body.mass_kg - rotor_mass
    if body_mass <= 0:
        raise ValueError(f"massa dei rotori nel USD ({rotor_mass:.3f} kg) >= massa totale "
                         f"({rigid_body.mass_kg:.3f} kg)")
    # baricentro del body: m_tot cg = m_body cg_body + sum m_r r
    first_moment = rigid_body.mass_kg * cg - sum(m * np.asarray(p) for m, p, _ in rotors)
    cg_body = first_moment / body_mass
    # inerzia del body attorno al baricentro TOTALE, poi trasportata sul suo baricentro
    inertia = total.copy()
    for m, p, own in rotors:
        inertia -= point_mass_inertia(m, np.asarray(p) - cg)
        if own is not None:
            inertia -= own
    inertia -= point_mass_inertia(body_mass, cg_body - cg)
    values, axes = principal(inertia)
    if np.any(values <= 0):
        raise ValueError(f"inerzia del body non definita positiva {values}: controlla "
                         "rigid_body.inertia_kg_m2 e le masse dei rotori nel USD")
    return MassProperties(body_mass, cg_body, values, axes)


def rotor_layout_mismatch(declared: list, actual: list, tolerance_m: float = 0.01) -> list[str]:
    """Differenze fra le posizioni dei rotori dichiarate (`rotors.positions_m`) e quelle del USD."""
    problems = []
    for i, (d, a) in enumerate(zip(declared, actual, strict=True)):
        delta = max(abs(float(x) - float(y)) for x, y in zip(d, a, strict=True))
        if delta > tolerance_m:
            problems.append(f"rotor{i}: config {tuple(round(float(v), 4) for v in d)} contro "
                            f"USD {tuple(round(float(v), 4) for v in a)}")
    return problems

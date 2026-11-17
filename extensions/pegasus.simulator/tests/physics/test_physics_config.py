"""Config, assi e masse della fisica avanzata: Python puro, senza fastuav e senza Isaac.

    <ISAACSIM_PATH>/python.sh -m pytest extensions/pegasus.simulator/tests/physics
"""

from __future__ import annotations

from math import isclose, pi

import numpy as np
import pytest

from pegasus.simulator.logic.physics.config import (
    BatteryConfig,
    InertiaConfig,
    PhysicsConfig,
    PropellerConfig,
    RigidBodyConfig,
    RotorLayoutConfig,
)
from pegasus.simulator.logic.physics.frames import (
    air_data,
    matrix_to_quat,
    quat_to_matrix,
    rot_y,
    rot_z,
)
from pegasus.simulator.logic.physics.rigid_body import (
    body_mass_properties,
    inertia_matrix,
    point_mass_inertia,
    rotor_layout_mismatch,
)


def test_from_dict_is_partial_and_nested() -> None:
    config = PhysicsConfig.from_dict({"motor": {"kv_rpm_per_v": 1040.0},
                                      "rigid_body": {"inertia_kg_m2": {"ixx": 0.1}}})
    assert config.motor.kv_rpm_per_v == 1040.0
    assert config.motor.resistance_ohm == PhysicsConfig().motor.resistance_ohm
    assert config.rigid_body.inertia_kg_m2.ixx == 0.1
    assert config.rigid_body.inertia_kg_m2.izz == InertiaConfig().izz


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="chiavi sconosciute"):
        PhysicsConfig.from_dict({"motore": {}})
    with pytest.raises(ValueError, match="chiavi sconosciute"):
        PhysicsConfig.from_dict({"motor": {"kv": 1000}})


def test_round_trip_through_dict() -> None:
    config = PhysicsConfig.from_dict({"battery": {"cells_series": 8}})
    assert PhysicsConfig.from_dict(config.to_dict()) == config
    assert PhysicsConfig.coerce(config) is config


def test_rotor_layout_needs_four_balanced_rotors() -> None:
    with pytest.raises(ValueError):
        RotorLayoutConfig(positions_m=[(0.0, 0.0, 0.0)] * 3, directions=["cw", "ccw", "cw"])
    with pytest.raises(ValueError):
        RotorLayoutConfig(directions=["cw", "cw", "cw", "ccw"])
    with pytest.raises(ValueError):
        RotorLayoutConfig(directions=["cw", "ccw", "cw", "orario"])


def test_propeller_models_need_twelve_coefficients() -> None:
    with pytest.raises(ValueError):
        PropellerConfig(ct_model=[0.1] * 10)
    assert PropellerConfig(ct_model=[0.0] * 12).ct_model == [0.0] * 12


def test_battery_voltages_are_ordered() -> None:
    with pytest.raises(ValueError):
        BatteryConfig(v_cut_v=3.8, v_mean_v=3.7)


@pytest.mark.parametrize("rpy", [(0.0, 0.0, 0.0), (0.3, -0.2, 1.0), (pi, 0.1, -2.0)])
def test_quaternion_round_trip(rpy: tuple[float, float, float]) -> None:
    r = rot_z(rpy[2]) @ rot_y(rpy[1])
    assert np.allclose(quat_to_matrix(matrix_to_quat(r)), r)


def test_air_data_tailsitter_angles() -> None:
    """Muso (+z) nel vento: incidenza 0. Hover con vento laterale: 90 gradi."""
    nose_forward = rot_y(pi / 2)  # +z corpo verso +x mondo
    cruise = air_data(np.array([20.0, 0.0, 0.0]), nose_forward, np.zeros(3))
    assert isclose(cruise.body_incidence, 0.0, abs_tol=1e-9)
    assert isclose(cruise.disk_alpha, pi / 2)
    hover = air_data(np.array([5.0, 0.0, 0.0]), np.eye(3), np.zeros(3))
    assert isclose(hover.body_incidence, pi / 2)
    assert np.allclose(hover.normal_dir, [1.0, 0.0, 0.0])
    still_in_wind = air_data(np.zeros(3), np.eye(3), np.array([-5.0, 0.0, 0.0]))
    assert np.allclose(still_in_wind.u, [5.0, 0.0, 0.0])


def test_body_mass_removes_rotors_exactly() -> None:
    """Body + rotori puntiformi ricompongono massa, baricentro e inerzia del totale."""
    rotors = [(0.05, np.array(p), None) for p in
              [(0.2, 0.0, 0.1), (0.0, 0.2, 0.1), (-0.2, 0.0, 0.1), (0.0, -0.2, 0.1)]]
    total = RigidBodyConfig(mass_kg=2.0, cg_m=(0.0, 0.0, 0.01),
                            inertia_kg_m2=InertiaConfig(ixx=0.05, iyy=0.05, izz=0.08, ixy=0.001))
    props = body_mass_properties(total, rotors)
    assert isclose(props.mass_kg, 1.8)
    cg = np.asarray(total.cg_m)
    rebuilt = props.inertia + point_mass_inertia(props.mass_kg, props.center_of_mass_m - cg)
    for m, p, _ in rotors:
        rebuilt += point_mass_inertia(m, p - cg)
    assert np.allclose(rebuilt, inertia_matrix(total.inertia_kg_m2))
    moment = props.mass_kg * props.center_of_mass_m + sum(m * p for m, p, _ in rotors)
    assert np.allclose(moment / total.mass_kg, cg)


def test_body_mass_rejects_impossible_values() -> None:
    rotors = [(0.6, np.array([0.2, 0.0, 0.0]), None)] * 4
    with pytest.raises(ValueError, match="massa dei rotori"):
        body_mass_properties(RigidBodyConfig(mass_kg=2.0), rotors)


def test_rotor_layout_mismatch() -> None:
    declared = RotorLayoutConfig().positions_m
    assert rotor_layout_mismatch(declared, declared) == []
    moved = [list(p) for p in declared]
    moved[2][2] += 0.05
    problems = rotor_layout_mismatch(declared, moved)
    assert len(problems) == 1 and problems[0].startswith("rotor2")

"""Modello aerodinamico a polari (aero.polar): tabella, Reynolds, raccordo, validazione."""

from __future__ import annotations

from dataclasses import replace
from math import isclose, pi, radians

import numpy as np
import pytest

from pegasus.simulator.logic.physics.config import (
    AeroConfig,
    PhysicsConfig,
    PolarCaseConfig,
    PolarConfig,
    WingConfig,
)

pytest.importorskip("fastuav.physics", reason="fastuav non installata")

from pegasus.simulator.logic.physics.aero import Airframe  # noqa: E402
from pegasus.simulator.logic.physics.frames import air_data, rot_y  # noqa: E402

RHO, NU = 1.1, 1.6e-5
ALPHA = [0.0, 5.0, 10.0, 15.0]
LOW = PolarCaseConfig("basso", 2.0e6, 12.0, ALPHA, [0.0, 0.4, 0.8, 1.0], [0.05, 0.06, 0.10, 0.20])
HIGH = PolarCaseConfig("alto", 4.0e6, 14.0, ALPHA, [0.0, 0.5, 1.0, 1.2], [0.04, 0.05, 0.08, 0.16])


def physics(*cases: PolarCaseConfig, blend: float = 5.0) -> PhysicsConfig:
    config = PhysicsConfig()
    config.atmosphere.rho_kg_m3 = RHO
    config.atmosphere.kinematic_viscosity_m2_s = NU
    config.aero = AeroConfig(polar=PolarConfig(enabled=True, s_ref_m2=0.03, blend_deg=blend,
                                               cases=list(cases)))
    return config


def airframe(config: PhysicsConfig) -> Airframe:
    return Airframe(config.aero, config.atmosphere, 0.0)


def load_at(frame: Airframe, speed: float, incidence_deg: float):
    """Volo nel piano x-z: muso inclinato in modo che l'incidenza sul vento sia quella data."""
    r = rot_y(pi / 2 - radians(incidence_deg))
    return frame.load(air_data(np.array([speed, 0.0, 0.0]), r, np.zeros(3)))


def test_grid_points_reproduce_the_polar() -> None:
    frame = airframe(physics(LOW))
    speed = 40.0
    q_s = 0.5 * RHO * speed**2 * 0.03
    for alpha, cl, cd in zip(LOW.alpha_deg[:3], LOW.cl, LOW.cd):
        load = load_at(frame, speed, alpha)
        assert isclose(load.lift_n, q_s * cl, rel_tol=1e-6, abs_tol=1e-9)
        assert isclose(load.drag_n, q_s * cd, rel_tol=1e-6)
        assert load.polar_weight == 1.0


def test_reynolds_interpolation_and_clamping() -> None:
    frame = airframe(physics(HIGH, LOW))  # ordine qualsiasi: si ordinano per Re
    table = frame.polar
    assert isclose(table.coefficients(5.0, 3.0e6)[0], 0.45)
    assert isclose(table.stall_deg(3.0e6), 13.0)
    assert table.coefficients(5.0, 1.0e5) == table.coefficients(5.0, 2.0e6)
    assert table.coefficients(5.0, 9.0e6) == table.coefficients(5.0, 4.0e6)


def test_blend_is_continuous_and_ends_on_body_model() -> None:
    config = physics(LOW)
    frame = airframe(config)
    body_only = airframe(replace(config, aero=AeroConfig(body=config.aero.body)))
    speed = 30.0
    stall, end = LOW.alpha_stall_deg, LOW.alpha_stall_deg + 5.0
    for edge in (stall, end):
        before, after = load_at(frame, speed, edge - 1e-4), load_at(frame, speed, edge + 1e-4)
        assert np.allclose(before.force_n, after.force_n, atol=1e-3)
    mid = load_at(frame, speed, stall + 2.5)
    assert isclose(mid.polar_weight, 0.5, abs_tol=1e-6)
    for incidence in (end + 1.0, 45.0, 90.0):
        assert np.allclose(load_at(frame, speed, incidence).force_n,
                           load_at(body_only, speed, incidence).force_n)


def test_disabled_polar_changes_nothing() -> None:
    config = physics(LOW)
    off = replace(config, aero=replace(config.aero, polar=replace(config.aero.polar, enabled=False)))
    plain = replace(config, aero=AeroConfig(body=config.aero.body))
    for incidence in (2.0, 10.0, 60.0):
        assert np.allclose(load_at(airframe(off), 25.0, incidence).force_n,
                           load_at(airframe(plain), 25.0, incidence).force_n)


def test_polar_config_validation() -> None:
    with pytest.raises(ValueError, match="stessa"):
        PolarCaseConfig("x", 1e6, 5.0, [0.0, 10.0], [0.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="crescente"):
        PolarCaseConfig("x", 1e6, 5.0, [0.0, 10.0, 10.0], [0.0, 1.0, 1.0], [0.1, 0.2, 0.2])
    with pytest.raises(ValueError, match="stallo|alpha_stall"):
        PolarCaseConfig("x", 1e6, 20.0, [0.0, 10.0], [0.0, 1.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="almeno un caso"):
        PolarConfig(enabled=True, s_ref_m2=0.03)
    with pytest.raises(ValueError, match="distinti"):
        PolarConfig(enabled=True, s_ref_m2=0.03, cases=[LOW, LOW])
    with pytest.raises(ValueError, match="insieme"):
        AeroConfig(wing=WingConfig(enabled=True), polar=PolarConfig(enabled=True, s_ref_m2=0.03,
                                                                    cases=[LOW]))


def test_from_dict_builds_polar_cases() -> None:
    config = PhysicsConfig.from_dict({"aero": {"polar": {
        "enabled": True, "s_ref_m2": 0.03,
        "cases": [{"name": "a", "reynolds_per_m": 1e6, "alpha_stall_deg": 5.0,
                   "alpha_deg": [0.0, 10.0], "cl": [0.0, 1.0], "cd": [0.1, 0.2]}]}}})
    assert isinstance(config.aero.polar.cases[0], PolarCaseConfig)
    assert config.aero.polar.cases[0].cl == [0.0, 1.0]


# ---------------------------------------------------------------- momento (Cm)

def with_moment(cm: list[float], c_ref: float = 0.1, ref_z: float = 0.0) -> PhysicsConfig:
    config = physics(replace(LOW, cm=cm))
    config.aero.polar = replace(config.aero.polar, moment_enabled=True, c_ref_m=c_ref,
                                moment_ref_z_m=ref_z)
    return config


def test_moment_is_off_by_default() -> None:
    load = load_at(airframe(physics(replace(LOW, cm=[0.1] * 4))), 30.0, 5.0)
    assert not np.any(load.torque_nm)


def test_moment_at_the_cg_is_q_s_c_cm() -> None:
    speed = 30.0
    load = load_at(airframe(with_moment([0.05] * 4)), speed, 5.0)
    expected = 0.5 * RHO * speed**2 * 0.03 * 0.1 * 0.05
    assert isclose(load.pitching_moment_nm, expected, rel_tol=1e-9)
    assert isclose(float(np.linalg.norm(load.torque_nm)), expected, rel_tol=1e-9)


def test_moment_reference_is_transported_to_the_cg() -> None:
    """Cm nullo al punto di riferimento 0,2 m davanti al baricentro: resta r x F."""
    frame = airframe(with_moment([0.0] * 4, ref_z=0.2))
    load = load_at(frame, 30.0, 5.0)
    assert np.allclose(load.torque_nm, np.cross([0.0, 0.0, 0.2], load.force_n))


def test_moment_config_validation() -> None:
    with pytest.raises(ValueError, match="lunghezza"):
        replace(LOW, cm=[0.0, 0.1])
    with pytest.raises(ValueError, match="c_ref_m"):
        PolarConfig(enabled=True, s_ref_m2=0.03, cases=[replace(LOW, cm=[0.0] * 4)],
                    moment_enabled=True)
    with pytest.raises(ValueError, match="mancano i cm"):
        PolarConfig(enabled=True, s_ref_m2=0.03, cases=[LOW], moment_enabled=True, c_ref_m=0.1)

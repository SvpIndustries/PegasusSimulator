"""Modello fisico avanzato su fastuav.physics: si salta se fastuav non è installata.

Usano i default generici: verificano la struttura e la coerenza fisica, non i valori di
un drone specifico.

    <ISAACSIM_PATH>/python.sh -m pytest extensions/pegasus.simulator/tests/physics
"""

from __future__ import annotations

import inspect
from math import isclose, pi

import numpy as np
import pytest

pytest.importorskip("fastuav.physics", reason="fastuav non installata")

from fastuav.physics import battery as fastuav_battery  # noqa: E402
from fastuav.physics.flight import MultirotorAero  # noqa: E402
from fastuav.physics.motor import MotorPerformanceModel  # noqa: E402
from fastuav.physics.propeller import PropellerAerodynamicsModel, TipSpeedLimit  # noqa: E402

from pegasus.simulator.logic.physics.bench import build  # noqa: E402
from pegasus.simulator.logic.physics.config import (  # noqa: E402
    AeroConfig,
    PhysicsConfig,
    WingConfig,
)
from pegasus.simulator.logic.physics.controller import GeometricControllerConfig  # noqa: E402
from pegasus.simulator.logic.physics.frames import BodyState, air_data, rot_y  # noqa: E402
from pegasus.simulator.logic.physics.model import AdvancedMultirotorPhysics  # noqa: E402
from pegasus.simulator.logic.physics.sim6dof import simulate  # noqa: E402

DT = 1 / 240


def flight(**controller: object):
    return build(PhysicsConfig(), GeometricControllerConfig(**controller))


def default_model() -> AdvancedMultirotorPhysics:
    return AdvancedMultirotorPhysics(PhysicsConfig())


# ------------------------------------------------------------------ API di fastuav usata


def test_fastuav_api_used_by_advanced_physics() -> None:
    """Se FAST-UAV cambia queste firme, il modello va aggiornato: meglio saperlo qui."""
    incidence = inspect.signature(PropellerAerodynamicsModel.aero_coefficients_incidence)
    assert {"beta", "J", "alpha", "n_blades", "chord_to_radius", "ct_model", "cp_model"} <= set(
        incidence.parameters)
    assert {"diameter", "speed_of_sound", "airspeed", "alpha", "mach_max"} <= set(
        inspect.signature(TipSpeedLimit).parameters)
    for name in ("current", "voltage", "power"):
        assert callable(getattr(MotorPerformanceModel, name))
    for name in ("pack_open_circuit_voltage", "terminal_voltage", "depth_of_discharge", "current"):
        assert callable(getattr(fastuav_battery, name))
    for name in ("lift", "drag", "pitching_moment"):
        assert callable(getattr(MultirotorAero, name))


# ------------------------------------------------------------------ componenti


def test_propeller_static_is_quadratic() -> None:
    model = default_model()
    k_t, k_q = model.propeller.static_constants()
    for omega in (300.0, 600.0, 900.0):
        load = model.propeller.load(omega, 0.0, 0.0)
        assert isclose(load.thrust_n, k_t * omega**2, rel_tol=1e-6)
    assert k_q > 0


def test_propeller_thrust_drops_with_axial_speed() -> None:
    model = default_model()
    static = model.propeller.load(800.0, 0.0, 0.0).thrust_n
    climbing = model.propeller.load(800.0, 15.0, 0.0).thrust_n
    assert climbing < static


def test_motor_follows_first_order() -> None:
    model = default_model()
    motor = model.motors[0]
    tau = PhysicsConfig().motor.time_constant_s
    motor.reset(0.0)
    t = 0.0
    while t < tau - 1e-12:
        motor.step(1000.0, 1e9, 0.0, DT / 10)
        t += DT / 10
    assert isclose(motor.omega, 1000.0 * (1 - np.exp(-1)), rel_tol=0.01)


def test_motor_speed_is_limited_by_voltage() -> None:
    model = default_model()
    motor = model.motors[0]
    limit = motor.speed_limit(14.8, 0.0)
    for _ in range(2000):
        motor.step(1e6, limit, 0.0, DT)
    assert motor.omega <= limit + 1e-6


def test_battery_sags_under_load() -> None:
    model = default_model()
    rest = model.battery.open_circuit_voltage()
    assert model.battery.step(500.0, DT) < rest


def test_hover_reaction_torques_cancel() -> None:
    model = default_model()
    w = model.hover_omega()
    model.reset(w)
    wrench = model.step(BodyState(), [w] * 4, DT)
    force, torque = model.total_about_cg(wrench)
    assert np.allclose(torque, 0.0, atol=1e-9)
    assert isclose(force[2], PhysicsConfig().rigid_body.mass_kg * 9.80665, rel_tol=0.02)


def test_aero_opposes_motion_and_vanishes_at_rest() -> None:
    model = default_model()
    rest = model.airframe.load(air_data(np.zeros(3), np.eye(3), np.zeros(3)))
    assert not np.any(rest.force_n)
    for velocity, r in [((10.0, 0.0, 0.0), np.eye(3)), ((0.0, 0.0, 10.0), np.eye(3)),
                        ((15.0, 0.0, 0.0), rot_y(1.2)), ((0.0, 0.0, -5.0), np.eye(3))]:
        v = np.array(velocity)
        load = model.airframe.load(air_data(v, r, np.zeros(3)))
        assert (r @ load.force_n) @ v < 0, f"la resistenza deve opporsi al moto {velocity}"


def test_wing_adds_lift_in_forward_flight() -> None:
    plain = default_model()
    winged = AdvancedMultirotorPhysics(
        PhysicsConfig(aero=AeroConfig(wing=WingConfig(enabled=True))))
    air = air_data(np.array([15.0, 0.0, 0.0]), rot_y(1.2), np.zeros(3))
    assert winged.airframe.load(air).lift_n > plain.airframe.load(air).lift_n


# ------------------------------------------------------------------ volo offline


def test_offline_hover_holds_setpoint() -> None:
    model, controller = flight(setpoint_m=(0.0, 0.0, 2.0))
    model.reset(model.hover_omega())
    trace = simulate(model, controller.update, BodyState(position=np.array([0.0, 0.0, 2.0])),
                     4.0, DT)
    assert np.linalg.norm(trace.positions[-1] - [0.0, 0.0, 2.0]) < 0.05
    assert trace.samples[-1].state.rotation[2, 2] > 0.999


def test_offline_takeoff_from_ground() -> None:
    model, controller = flight(setpoint_m=(1.0, -1.0, 3.0))
    trace = simulate(model, controller.update, BodyState(), 8.0, DT)
    assert np.linalg.norm(trace.positions[-1] - [1.0, -1.0, 3.0]) < 0.1


def test_offline_transition_tilts_and_accelerates() -> None:
    model, controller = flight(mode="transition", setpoint_m=(0.0, 0.0, 10.0),
                               transition_start_s=1.0, transition_duration_s=2.0,
                               transition_pitch_rad=0.6)
    model.reset(model.hover_omega())
    trace = simulate(model, controller.update, BodyState(position=np.array([0.0, 0.0, 10.0])),
                     5.0, DT)
    final = trace.samples[-1].state
    assert isclose(np.arccos(final.rotation[2, 2]), 0.6, abs_tol=0.05)
    assert final.linear_velocity[0] > 3.0
    assert abs(final.position[2] - 10.0) < 1.0


def test_rotor_speed_never_exceeds_tip_mach() -> None:
    model = default_model()
    w_mach = model.propeller.max_speed(0.0, 0.0)
    for _ in range(500):
        wrench = model.step(BodyState(), [1e6] * 4, DT)
    assert np.all(wrench.rotor_omega_rad_s <= w_mach * 1.0001)
    assert w_mach < 2 * pi * 1e5

"""Gemello offline: prove della fisica senza Isaac (stesso modello del Multirotor).

Casi: hover, step (gradino di giri), trim (volo livellato), transition (tailsitter),
bench (confronto con un riepilogo di banco). Si chiamano con `run_case`; HIL li espone
come CLI confiGOD (`hil-physics --config <scena> --case hover`).
"""

from __future__ import annotations

import csv
from dataclasses import replace
from math import degrees, hypot, pi, radians
from pathlib import Path
from typing import Any

import numpy as np

from pegasus.simulator.logic.physics.config import PhysicsConfig
from pegasus.simulator.logic.physics.controller import (
    GeometricController,
    GeometricControllerConfig,
    build_controller,
)
from pegasus.simulator.logic.physics.frames import E3, BodyState, air_data, rot_y
from pegasus.simulator.logic.physics.model import AdvancedMultirotorPhysics
from pegasus.simulator.logic.physics.motor import RPM_TO_RAD_S
from pegasus.simulator.logic.physics.sim6dof import Trace, simulate

G = 9.80665
CASES = ("hover", "step", "trim", "transition", "bench")


def run_case(case: str, physics: PhysicsConfig, controller: GeometricControllerConfig,
             dt: float, *, seconds: float = 10.0, speeds: str = "5,10,15,20",
             bench_file: str = "", out: Path | None = None) -> int:
    """Esegue un caso e stampa il riepilogo; restituisce il codice di uscita."""
    runners = {"hover": _hover, "step": _step, "trim": _trim,
               "transition": _transition, "bench": _bench}
    if case not in runners:
        raise ValueError(f"caso {case!r} sconosciuto: {CASES}")
    options = {"seconds": seconds, "speeds": speeds, "bench_file": bench_file}
    _describe(physics, controller)
    return runners[case](physics, controller, dt, options, out)


def build(physics: PhysicsConfig, controller: GeometricControllerConfig
          ) -> tuple[AdvancedMultirotorPhysics, GeometricController]:
    model = AdvancedMultirotorPhysics(physics)
    return model, build_controller(controller, model)


# ---------------------------------------------------------------------------- casi


def _describe(physics: PhysicsConfig, controller: GeometricControllerConfig) -> None:
    model, _ = build(physics, controller)
    k_t, k_q = model.propeller.static_constants()
    print(f"modello: massa {model.mass_kg:.3f} kg | elica D={physics.propeller.diameter_m} m "
          f"beta={physics.propeller.beta} | k_T={k_t:.3e} N/(rad/s)^2 "
          f"k_Q={k_q:.3e} Nm/(rad/s)^2 | batteria {model.battery.open_circuit_voltage():.2f} V")


def _hover(physics: PhysicsConfig, controller: GeometricControllerConfig, dt: float,
           options: dict[str, Any], out: Path | None) -> int:
    model, ctrl = build(physics, controller)
    w_hover = model.hover_omega()
    model.reset(w_hover)
    start = BodyState(position=np.asarray(controller.setpoint_m, dtype=float))
    trace = simulate(model, ctrl.update, start, options["seconds"], dt)
    last = trace.samples[len(trace.samples) // 2:]
    omega = np.mean([s.wrench.rotor_omega_rad_s for s in last], axis=0)
    thrust = np.mean([s.wrench.rotor_thrust_n for s in last], axis=0)
    power = float(np.mean([s.wrench.telemetry["bus_power"] for s in last]))
    current = float(np.mean([s.wrench.telemetry["bus_current"] for s in last]))
    volts = trace.samples[-1].wrench.telemetry["bus_voltage"]
    error = np.linalg.norm(trace.positions[-1] - np.asarray(controller.setpoint_m))
    tilt = max(np.degrees(np.arccos(np.clip(s.state.rotation[2, 2], -1, 1))) for s in last)
    energy = model.battery.energy_j * physics.battery.max_dod
    print(f"hover: giri {omega.mean() / RPM_TO_RAD_S:.0f} rpm (stima statica "
          f"{w_hover / RPM_TO_RAD_S:.0f}) | spinta/rotore {thrust.mean():.2f} N "
          f"(peso/4 {model.mass_kg * G / 4:.2f} N)")
    print(f"       batteria {volts:.2f} V, {current:.1f} A, {power:.0f} W | autonomia stimata "
          f"{energy / power / 60 if power > 0 else float('inf'):.1f} min")
    print(f"       errore di posizione finale {error:.3f} m | inclinazione max {tilt:.2f} gradi")
    _save(trace, out, "hover.csv")
    return 0 if error < 0.2 and tilt < 5.0 else 1


def _step(physics: PhysicsConfig, controller: GeometricControllerConfig, dt: float,
           options: dict[str, Any], out: Path | None) -> int:
    """Gradino di giri di un motore: dal valore di hover a +10%."""
    model, _ = build(physics, controller)
    motor = model.motors[0]
    w0 = model.hover_omega()
    motor.reset(w0)
    w1 = 1.1 * w0
    t, times, values = 0.0, [], []
    while t < 10 * physics.motor.time_constant_s:
        motor.step(w1, 1e9, 0.0, dt)
        t += dt
        times.append(t)
        values.append(motor.omega)
    reached = next(t for t, w in zip(times, values, strict=True) if w >= w0 + 0.632 * (w1 - w0))
    print(f"gradino: {w0 / RPM_TO_RAD_S:.0f} -> {w1 / RPM_TO_RAD_S:.0f} rpm, 63% in "
          f"{reached * 1000:.1f} ms (tau = {physics.motor.time_constant_s * 1000:.1f} ms, "
          f"risoluzione {dt * 1000:.1f} ms)")
    return 0


def _trim(physics: PhysicsConfig, controller: GeometricControllerConfig, dt: float,
           options: dict[str, Any], out: Path | None) -> int:
    """Volo livellato a velocità costante nel piano x-z: inclinazione e giri di equilibrio.

    Con un'ala gli equilibri possono essere due (ala portante a bassa incidenza, corpo di
    traverso ad alta incidenza): si parte sia dalla soluzione precedente sia da un assetto
    di crociera, e fra gli equilibri trovati si tiene quello di potenza minima.
    """
    from scipy.optimize import least_squares

    model, _ = build(physics, controller)
    speeds = [float(v) for v in options["speeds"].split(",")]
    rows = []
    print(" V [m/s] | incl. [gradi] | alpha [gradi] | giri [rpm] | spinta tot [N] | potenza [W]"
          " | L [N] | D [N] | momento aero [Nm]")
    guess = np.array([0.3, model.hover_omega()])
    cruise_guess = np.array([radians(80.0), model.hover_omega() * 0.6])
    tolerance = 0.01 * model.mass_kg * G
    for speed in speeds:
        def residual(x: np.ndarray, speed: float = speed) -> np.ndarray:
            force, _ = _level_flight_forces(model, speed, x[0], x[1])
            return force[[0, 2]] / (model.mass_kg * G)

        best = None
        for start in (guess, cruise_guess):
            sol = least_squares(residual, start, bounds=([0.0, 1.0], [pi / 2, 1e5]))
            force, info = _level_flight_forces(model, speed, *sol.x)
            ok = bool(np.linalg.norm(force[[0, 2]]) < tolerance)
            key = (not ok, info["power"])
            if best is None or key < best[0]:
                best = (key, sol, info, ok)
        _, sol, info, ok = best
        pitch, omega = sol.x
        print(f" {speed:7.1f} | {degrees(pitch):13.1f} | {90.0 - degrees(pitch):13.2f} | "
              f"{omega / RPM_TO_RAD_S:10.0f} | {info['thrust']:14.2f} | {info['power']:11.0f} | "
              f"{info['lift']:5.2f} | {info['drag']:5.2f} | "
              f"{info['moment']:8.3f}{'' if ok else '  (non in equilibrio)'}")
        rows.append({"speed": speed, "pitch_deg": degrees(pitch), "alpha_deg": 90.0 - degrees(pitch),
                     "rpm": omega / RPM_TO_RAD_S, **info, "converged": ok})
        guess = sol.x
    _save_rows(rows, out, "trim.csv")
    return 0


def _level_flight_forces(model: AdvancedMultirotorPhysics, speed: float, pitch: float,
                         omega: float) -> tuple[np.ndarray, dict[str, float]]:
    """Forza risultante (mondo) a velocità (speed, 0, 0), muso inclinato di `pitch` verso +x."""
    r = rot_y(pitch)
    air = air_data(np.array([speed, 0.0, 0.0]), r, model.wind)
    load = model.propeller.load(omega, float(air.u[2]), hypot(air.u[0], air.u[1]))
    aero = model.airframe.load(air)
    force_b = aero.force_n + 4 * load.thrust_n * E3
    bus_v = model.battery.open_circuit_voltage()
    electrics = model.motors[0].electrics(load.torque_nm, omega, bus_v)
    force = r @ force_b + model.mass_kg * np.array([0.0, 0.0, -G])
    return force, {"thrust": 4 * load.thrust_n, "power": 4 * electrics.bus_power_w,
                   "lift": aero.lift_n, "drag": aero.drag_n, "moment": aero.pitching_moment_nm,
                   "J": load.advance_ratio}


def _transition(physics: PhysicsConfig, controller: GeometricControllerConfig, dt: float,
           options: dict[str, Any], out: Path | None) -> int:
    if controller.mode != "transition":
        print("nota: il controllore non è in modalità 'transition', uso comunque la rampa di assetto")
        controller = replace(controller, mode="transition")
    model, ctrl = build(physics, controller)
    model.reset(model.hover_omega())
    start = BodyState(position=np.asarray(controller.setpoint_m, dtype=float))
    trace = simulate(model, ctrl.update, start, options["seconds"], dt)
    final = trace.samples[-1]
    z0 = controller.setpoint_m[2]
    pitch = degrees(np.arccos(np.clip(final.state.rotation[2, 2], -1, 1)))
    print(f"transizione: velocità finale {np.linalg.norm(final.state.linear_velocity):.1f} m/s, "
          f"inclinazione {pitch:.0f} gradi, quota {final.state.position[2]:.2f} m "
          f"(partenza {z0:.2f}), min {trace.positions[:, 2].min():.2f} m")
    print(f"             potenza {final.wrench.telemetry['bus_power']:.0f} W, portanza "
          f"{final.wrench.telemetry['lift']:.1f} N, "
          f"resistenza {final.wrench.telemetry['drag']:.1f} N")
    _save(trace, out, "transition.csv")
    return 0


def _bench(physics: PhysicsConfig, controller: GeometricControllerConfig, dt: float,
           options: dict[str, Any], out: Path | None) -> int:
    """Confronto con un riepilogo di banco (run_*_summary.csv): spinta e corrente contro giri."""
    path = options["bench_file"]
    if not path:
        print("serve --bench-file <run_*_summary.csv>")
        return 2
    model, _ = build(physics, controller)
    rows = [r for r in _read_bench(Path(path)) if float(r["esc_rpm"] or 0) > 0]
    print(" rpm    | spinta banco [N] | modello [N] | err % | corrente banco [A] | modello [A]")
    errors = []
    for r in rows:
        omega = float(r["esc_rpm"]) * RPM_TO_RAD_S
        load = model.propeller.load(omega, 0.0, 0.0)
        electrics = model.motors[0].electrics(load.torque_nm, omega, float(r["bus_v"]))
        measured = float(r["thrust_g"]) * G / 1000.0
        err = 100 * (load.thrust_n - measured) / measured if measured > 0.1 else float("nan")
        errors.append(err)
        print(f" {float(r['esc_rpm']):6.0f} | {measured:16.2f} | {load.thrust_n:11.2f} | "
              f"{err:5.1f} | {float(r['current_a']):18.1f} | {electrics.current_a:11.1f}")
    valid = [e for e in errors if e == e]
    if valid:
        print(f"errore medio di spinta {np.mean(valid):.1f}% (|max| {np.max(np.abs(valid)):.1f}%)")
    return 0


def _read_bench(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


# ---------------------------------------------------------------------------- uscite


def _save(trace: Trace, out: Path | None, name: str) -> None:
    if out is None:
        return
    rows = []
    for s in trace.samples:
        row: dict[str, float] = {"t": s.t}
        for axis, value in zip("xyz", s.state.position, strict=True):
            row[f"p{axis}"] = float(value)
        for axis, value in zip("xyz", s.state.linear_velocity, strict=True):
            row[f"v{axis}"] = float(value)
        row["tilt_deg"] = degrees(np.arccos(np.clip(s.state.rotation[2, 2], -1, 1)))
        row.update(s.wrench.telemetry)
        rows.append(row)
    _save_rows(rows, out, name)


def _save_rows(rows: list[dict[str, Any]], out: Path | None, name: str) -> None:
    if out is None or not rows:
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"scritto {out / name}")


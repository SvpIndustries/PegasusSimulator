"""Motore + ESC: dal comando di giri ai giri reali, con i limiti elettrici.

Delega a FAST-UAV:
- `fastuav.physics.motor.MotorPerformanceModel` (corrente, tensione, potenza);
- `fastuav.physics.esc.ESCPerformanceModel` (rendimento dell'ESC).

Il comando è la velocità angolare desiderata [rad/s], la convenzione dei backend di
Pegasus (`input_reference`). I giri reali la inseguono con un primo ordine (tau),
limitati dalla tensione disponibile (Kv·(U_batt - R·I)) e dalla coppia massima.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt

from fastuav.physics.esc import ESCPerformanceModel
from fastuav.physics.motor import MotorPerformanceModel

from pegasus.simulator.logic.physics.config import EscConfig, MotorConfig

RPM_TO_RAD_S = 2.0 * pi / 60.0


@dataclass(frozen=True)
class MotorElectrics:
    current_a: float
    voltage_v: float
    power_w: float  # potenza elettrica al motore
    esc_efficiency: float
    bus_power_w: float  # potenza chiesta alla batteria


class Motor:
    def __init__(self, motor: MotorConfig, esc: EscConfig) -> None:
        self.s = motor
        self.esc = esc
        self.kv = motor.kv_rpm_per_v * RPM_TO_RAD_S  # (rad/s)/V, come lo usa FAST-UAV
        self.omega_min = motor.min_rpm * RPM_TO_RAD_S
        self.omega = 0.0

    def reset(self, omega: float = 0.0) -> None:
        self.omega = omega

    def speed_limit(self, bus_voltage: float, current: float) -> float:
        """Giri massimi a vuoto con la tensione al bus e la caduta resistiva."""
        return max(self.kv * (bus_voltage - self.s.resistance_ohm * current), 0.0)

    def step(self, omega_cmd: float, omega_max: float, torque_at_omega: float, dt: float) -> float:
        """Avanza i giri di dt verso il comando, limitato a [omega_min, omega_max].

        `torque_at_omega` è la coppia d'elica ai giri attuali: se supera la coppia massima,
        si abbassa il limite (la coppia cresce circa con w²).
        """
        limit = omega_max
        if self.s.max_torque_nm > 0 and torque_at_omega > self.s.max_torque_nm and self.omega > 0:
            limit = min(limit, self.omega * sqrt(self.s.max_torque_nm / torque_at_omega))
        target = min(max(omega_cmd, self.omega_min), max(limit, self.omega_min))
        if omega_cmd <= 0.0:
            target = 0.0  # disarmato
        self.omega += (target - self.omega) * (1.0 - exp(-dt / self.s.time_constant_s))
        return self.omega

    def electrics(self, torque: float, omega: float, bus_voltage: float) -> MotorElectrics:
        if omega <= 1e-6:
            return MotorElectrics(0.0, 0.0, 0.0, 1.0, 0.0)
        current = MotorPerformanceModel.current(torque, self.s.friction_torque_nm, self.kv)
        voltage = MotorPerformanceModel.voltage(current, omega, self.s.resistance_ohm, self.kv)
        power = MotorPerformanceModel.power(voltage, current)
        eta = 1.0
        if self.esc.enabled and power > 0:
            e = self.esc
            eta = ESCPerformanceModel.efficiency(
                power, current, bus_voltage, e.mosfet_resistance_ohm, e.diode_voltage_v,
                e.dead_time_s, e.pwm_frequency_hz, e.switching_coefficient, e.fixed_losses_w,
                eta_min=e.min_efficiency,
            )
        return MotorElectrics(float(current), float(voltage), float(power), float(eta),
                              float(power) / float(eta))

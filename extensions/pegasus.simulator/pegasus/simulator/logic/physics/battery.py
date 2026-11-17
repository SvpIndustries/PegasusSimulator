"""Batteria: tensione a vuoto secondo la scarica e tensione ai morsetti sotto carico.

Delega a `fastuav.physics.battery` (pack_open_circuit_voltage, terminal_voltage,
depth_of_discharge, current).
"""

from __future__ import annotations

from fastuav.physics import battery as fb

from pegasus.simulator.logic.physics.config import BatteryConfig


class Battery:
    def __init__(self, settings: BatteryConfig) -> None:
        self.s = settings
        self.nominal_v = settings.cells_series * settings.v_mean_v
        self.energy_j = self.nominal_v * settings.capacity_mah / 1000.0 * 3600.0
        self.reset()

    def reset(self) -> None:
        self.energy_drawn_j = self.s.initial_dod * self.energy_j
        self.voltage_v = self.open_circuit_voltage()
        self.current_a = 0.0

    @property
    def depth_of_discharge(self) -> float:
        return fb.depth_of_discharge(self.energy_drawn_j, self.energy_j, self.s.max_dod)

    def open_circuit_voltage(self) -> float:
        if not self.s.enabled:
            return self.nominal_v
        s = self.s
        return fb.pack_open_circuit_voltage(self.nominal_v, self.depth_of_discharge,
                                            s.v_full_v, s.v_mean_v, s.v_cut_v)

    def step(self, power_w: float, dt: float) -> float:
        """Eroga `power_w` per dt; restituisce la tensione ai morsetti."""
        if not self.s.enabled:
            self.voltage_v = self.nominal_v
        else:
            self.voltage_v = float(fb.terminal_voltage(self.open_circuit_voltage(),
                                                       self.s.resistance_ohm, power_w))
        self.current_a = fb.current(power_w, self.voltage_v)
        self.energy_drawn_j += max(power_w, 0.0) * dt
        return self.voltage_v

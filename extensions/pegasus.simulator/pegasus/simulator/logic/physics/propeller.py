"""Elica: spinta e coppia dai coefficienti di FAST-UAV, con flusso assiale e incidenza.

Delega a `fastuav.physics.propeller`:
- `PropellerAerodynamicsModel.aero_coefficients_incidence` per Ct, Cp(beta, J, alpha);
- `TipSpeedLimit` per il Mach alla punta (cap dei giri e penalità sul Cp).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, pi

import numpy as np
from fastuav.physics.propeller import PropellerAerodynamicsModel, TipSpeedLimit

from pegasus.simulator.logic.physics.config import AtmosphereConfig, PropellerConfig

#: oltre questo advance ratio il polinomio di FAST-UAV va in overflow (come in propeller_at_speed)
J_CAP = 4.0


@dataclass(frozen=True)
class RotorLoad:
    thrust_n: float
    torque_nm: float  # coppia aerodinamica sull'elica (positiva, si oppone alla rotazione)
    power_w: float
    advance_ratio: float
    disk_alpha_rad: float
    mach_tip: float


class Propeller:
    def __init__(self, settings: PropellerConfig, atmosphere: AtmosphereConfig) -> None:
        self.s = settings
        self.atm = atmosphere
        self._kwargs: dict[str, object] = {
            "n_blades": settings.n_blades,
            "chord_to_radius": settings.chord_to_radius,
        }
        if settings.ct_model is not None:
            self._kwargs["ct_model"] = np.array(settings.ct_model)
        if settings.cp_model is not None:
            self._kwargs["cp_model"] = np.array(settings.cp_model)

    def coefficients(self, advance_ratio: float, disk_alpha: float) -> tuple[float, float]:
        ct, cp = PropellerAerodynamicsModel.aero_coefficients_incidence(
            self.s.beta, advance_ratio, disk_alpha, **self._kwargs
        )
        return float(np.ravel(ct)[0]), float(np.ravel(cp)[0])

    def static_constants(self) -> tuple[float, float]:
        """(k_T, k_Q) equivalenti a punto fisso: T = k_T w², Q = k_Q w² (w in rad/s)."""
        ct, cp = self.coefficients(0.0, pi / 2.0)
        rho, d = self.atm.rho_kg_m3, self.s.diameter_m
        return ct * rho * d**4 / (2 * pi) ** 2, cp * rho * d**5 / (2 * pi) ** 3

    def max_speed(self, v_axial: float, v_edge: float) -> float:
        """Giri massimi [rad/s] per il Mach alla punta, nello stato di volo dato."""
        airspeed, alpha = hypot(v_axial, v_edge), atan2(abs(v_axial), v_edge)
        limit = TipSpeedLimit(self.s.diameter_m, self.atm.speed_of_sound_m_s, airspeed, alpha,
                              mach_max=self.s.mach_tip_max)
        return float(limit.max_speed)

    def load(self, omega: float, v_axial: float, v_edge: float) -> RotorLoad:
        """Carico dell'elica ai giri `omega` [rad/s].

        v_axial: velocità del disco rispetto all'aria lungo +z del rotore (positiva quando
                 l'aria entra dal davanti, cioè in salita/volo avanzato del tailsitter);
        v_edge:  componente nel piano del disco (>= 0).
        In discesa assiale (v_axial < 0) il modello di FAST-UAV non è valido (stato di
        anello vorticoso): si usano i coefficienti a flusso assiale nullo.
        """
        omega = max(float(omega), 0.0)
        if omega < 1e-3:
            return RotorLoad(0.0, 0.0, 0.0, 0.0, pi / 2.0, 0.0)
        v_ax = max(float(v_axial), 0.0)
        v_ed = abs(float(v_edge))
        airspeed = hypot(v_ax, v_ed)
        alpha = atan2(v_ax, v_ed) if airspeed > 1e-6 else pi / 2.0
        d, rho = self.s.diameter_m, self.atm.rho_kg_m3
        n = omega / (2.0 * pi)
        j = min(airspeed / (n * d), J_CAP)
        ct, cp = self.coefficients(j, alpha)
        limit = TipSpeedLimit(d, self.atm.speed_of_sound_m_s, airspeed, alpha,
                              mach_max=self.s.mach_tip_max)
        if self.s.compressibility:
            cp = float(np.ravel(limit.penalized_cp(cp, omega))[0])
        thrust = max(ct, 0.0) * rho * n**2 * d**4
        power = cp * rho * n**3 * d**5
        return RotorLoad(thrust, power / omega, power, j, alpha, float(limit.mach(omega)))

"""Aerodinamica dell'airframe: `MultirotorAero` di FAST-UAV portato in 3D.

FAST-UAV descrive il velivolo nel piano di beccheggio: dati (V, alpha) restituisce
portanza, resistenza e momento di beccheggio di fusoliera, bracci-ala e spinner, con il
corpo di rivoluzione lungo l'asse dei rotori (tailsitter). Qui:

1. il vento relativo si scompone in assi corpo (`frames.air_data`): componente lungo +z
   e componente normale, di direzione qualsiasi nel piano xy;
2. si chiedono a MultirotorAero L e D all'incidenza aJ, e si riportano in forza NORMALE
   N (opposta alla componente normale del moto) e ASSIALE A (opposta a quella assiale);
3. il momento di beccheggio agisce attorno all'asse normale al piano (z, u_normale).

Ipotesi: airframe assialsimmetrico attorno a z (vale per la "+" con 4 bracci uguali,
dove l'orientamento del piano di beccheggio rispetto ai bracci conta poco). Con la coda
nel vento (aJ > 90°, discesa) si usano gli stessi coefficienti a pi - aJ e il momento è
posto a zero: le stazioni sono misurate dal muso e il modello di FAST-UAV non copre il caso.

Polari (`aero.polar`, opzionali): CL(alpha), CD(alpha) dell'airframe completo, con alpha =
aJ (incidenza dell'asse del muso sul vento), interpolate sul Reynolds al metro V/nu. Fino
allo stallo sostituiscono L e D di MultirotorAero; fra stallo e stallo + blend_deg le forze
si raccordano linearmente a quelle del modello di corpo, che vale da solo oltre il raccordo
e con la coda nel vento. Il momento di beccheggio resta quello delle stazioni.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, degrees, pi, sin

import numpy as np
from fastuav.physics.flight import MultirotorAero

from pegasus.simulator.logic.physics.config import AeroConfig, AtmosphereConfig, PolarConfig
from pegasus.simulator.logic.physics.frames import E3, AirData


@dataclass(frozen=True)
class AeroLoad:
    force_n: np.ndarray  # assi corpo, applicata al baricentro
    torque_nm: np.ndarray  # assi corpo, attorno al baricentro
    lift_n: float
    drag_n: float
    pitching_moment_nm: float
    polar_weight: float = 0.0  # 1 = solo polare, 0 = solo modello di corpo


ZERO = AeroLoad(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0)


class PolarTable:
    """Polari dell'airframe interpolate in alpha (per caso) e poi nel Reynolds (fra casi)."""

    def __init__(self, polar: PolarConfig) -> None:
        self.cases = sorted(polar.cases, key=lambda c: c.reynolds_per_m)
        self.reynolds = np.array([c.reynolds_per_m for c in self.cases])
        self.s_ref_m2 = polar.s_ref_m2
        self.blend_deg = polar.blend_deg

    def _between(self, reynolds_per_m: float) -> tuple[int, int, float]:
        """Indici dei due casi attorno al Reynolds e peso del secondo (estremi: il più vicino)."""
        if len(self.cases) == 1 or reynolds_per_m <= self.reynolds[0]:
            return 0, 0, 0.0
        if reynolds_per_m >= self.reynolds[-1]:
            last = len(self.cases) - 1
            return last, last, 0.0
        j = int(np.searchsorted(self.reynolds, reynolds_per_m))
        i = j - 1
        return i, j, float((reynolds_per_m - self.reynolds[i]) / (self.reynolds[j] - self.reynolds[i]))

    def stall_deg(self, reynolds_per_m: float) -> float:
        i, j, w = self._between(reynolds_per_m)
        return (1.0 - w) * self.cases[i].alpha_stall_deg + w * self.cases[j].alpha_stall_deg

    def coefficients(self, alpha_deg: float, reynolds_per_m: float) -> tuple[float, float]:
        """(CL, CD) ad alpha [gradi], con alpha limitato alla griglia di ogni caso."""
        i, j, w = self._between(reynolds_per_m)
        values = []
        for case in (self.cases[i], self.cases[j]):
            values.append((float(np.interp(alpha_deg, case.alpha_deg, case.cl)),
                           float(np.interp(alpha_deg, case.alpha_deg, case.cd))))
        (cl_i, cd_i), (cl_j, cd_j) = values
        return (1.0 - w) * cl_i + w * cl_j, (1.0 - w) * cd_i + w * cd_j


def build_multirotor_aero(s: AeroConfig) -> MultirotorAero:
    body = s.body
    wing = None
    if s.wing.enabled:
        w = s.wing
        wing = {"S_single": w.area_single_m2, "AR": w.aspect_ratio, "n_lift": w.n_lift,
                "n_drag": w.n_drag, "CL_max": w.cl_max, "Cd0_wing": w.cd0_wing, "e": w.oswald}
    spinner = None
    if s.spinner.enabled:
        spinner = {"N": s.spinner.count, "d_spinner": s.spinner.diameter_m,
                   "Cd_spinner": s.spinner.cd}
    stations = None
    if s.stations.enabled:
        st = s.stations
        stations = {"body_length": st.body_length_m, "body_volume_ratio": st.body_volume_ratio,
                    "body_planform_centroid": st.body_planform_centroid_m,
                    "wing": st.wing_m, "spinner": st.spinner_m}
    return MultirotorAero(
        body.cd0, body.s_front_m2, body.s_top_m2, {"eta": body.eta, "Cd_n": body.cd_crossflow},
        C_L0=body.cl0, spinner=spinner, wing=wing, stations=stations,
    )


class Airframe:
    def __init__(self, settings: AeroConfig, atmosphere: AtmosphereConfig,
                 cg_z_m: float) -> None:
        self.s = settings
        self.rho = atmosphere.rho_kg_m3
        self.nu = atmosphere.kinematic_viscosity_m2_s
        self.model = build_multirotor_aero(settings) if settings.enabled else None
        self.polar = PolarTable(settings.polar) if settings.enabled and settings.polar.enabled else None
        # ascissa del baricentro dal muso, positiva verso la coda
        self.x_cg_from_nose = settings.stations.nose_z_m - cg_z_m

    def polar_forces(self, v: float, a_j: float) -> tuple[float, float, float] | None:
        """(L, D, peso della polare) dalle polari, o None se la polare non è attiva."""
        if self.polar is None:
            return None
        a_deg = degrees(a_j)
        reynolds = v / self.nu
        stall = self.polar.stall_deg(reynolds)
        weight = min(max(1.0 - (a_deg - stall) / self.polar.blend_deg, 0.0), 1.0)
        if weight <= 0.0:
            return 0.0, 0.0, 0.0
        cl, cd = self.polar.coefficients(min(a_deg, stall), reynolds)
        q_s = 0.5 * self.rho * v * v * self.polar.s_ref_m2
        return q_s * cl, q_s * cd, weight

    def load(self, air: AirData) -> AeroLoad:
        if self.model is None or air.airspeed < self.s.min_airspeed_m_s:
            return ZERO
        a_j = air.body_incidence
        tail_first = a_j > pi / 2.0
        if tail_first:
            a_j = pi - a_j
        alpha = pi / 2.0 - a_j
        v = air.airspeed
        lift = float(np.ravel(self.model.lift(v, alpha, self.rho))[0])
        drag = float(np.ravel(self.model.drag(v, alpha, self.rho))[0])
        weight = 0.0
        polar = None if tail_first else self.polar_forces(v, a_j)
        if polar is not None:
            polar_lift, polar_drag, weight = polar
            lift = weight * polar_lift + (1.0 - weight) * lift
            drag = weight * polar_drag + (1.0 - weight) * drag
        # (L, D) in assi vento -> (N, A) in assi corpo: inversa della rotazione di body_forces
        normal = lift * cos(a_j) + drag * sin(a_j)
        axial = drag * cos(a_j) - lift * sin(a_j)
        axial_sign = -1.0 if air.u[2] >= 0 else 1.0
        force = -normal * air.normal_dir + axial_sign * axial * E3
        moment = 0.0
        torque = np.zeros(3)
        if self.s.stations.enabled and not tail_first and np.any(air.normal_dir):
            moment = self.model.pitching_moment(v, alpha, self.rho, self.x_cg_from_nose)
            # positivo = aumenta l'incidenza: ruota +z lontano da u, verso -u_normale
            torque = moment * np.cross(air.normal_dir, E3)
        return AeroLoad(force, torque, lift, drag, moment, weight)

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
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
from fastuav.physics.flight import MultirotorAero

from pegasus.simulator.logic.physics.frames import E3, AirData
from pegasus.simulator.logic.physics.config import AeroConfig, AtmosphereConfig


@dataclass(frozen=True)
class AeroLoad:
    force_n: np.ndarray  # assi corpo, applicata al baricentro
    torque_nm: np.ndarray  # assi corpo, attorno al baricentro
    lift_n: float
    drag_n: float
    pitching_moment_nm: float


ZERO = AeroLoad(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0)


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
        self.model = build_multirotor_aero(settings) if settings.enabled else None
        # ascissa del baricentro dal muso, positiva verso la coda
        self.x_cg_from_nose = settings.stations.nose_z_m - cg_z_m

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
        return AeroLoad(force, torque, lift, drag, moment)

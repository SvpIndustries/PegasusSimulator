"""Parametri della fisica avanzata del multirotore (MultirotorConfig.physics).

Dataclass con i default nel codice, Python puro: nessun import di Isaac, di fastuav o di
librerie di configurazione. Chi usa Pegasus da solo passa un dict (anche parziale, anche
da YAML) a `PhysicsConfig.from_dict`; HIL valida i parametri con il suo schema confiGOD,
che ha gli stessi nomi di campo, e passa il dict risultante.

I default sono quelli di un quadricottero GENERICO (10", 4S, 2,5 kg), plausibili ma non
di un drone specifico: servono a far girare e provare il modello.

Convenzioni: corpo FLU (x avanti, y sinistra, z su = asse di spinta dei rotori, che per
un tailsitter è anche l'asse della fusoliera, muso verso +z); posizioni in metri rispetto
all'origine del prim `body`; mondo ENU.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

Vec3 = tuple[float, float, float]


# ----------------------------------------------------------------------------- corpo rigido


@dataclass
class InertiaConfig:
    """Tensore d'inerzia del drone COMPLETO attorno al baricentro, assi corpo, in kg·m²."""

    ixx: float = 0.035  # inerzia attorno a X
    iyy: float = 0.035  # inerzia attorno a Y
    izz: float = 0.060  # inerzia attorno a Z
    ixy: float = 0.0  # prodotto d'inerzia XY (elemento del tensore)
    ixz: float = 0.0
    iyz: float = 0.0


@dataclass
class RigidBodyConfig:
    """Massa, baricentro e inerzia del drone completo (batteria e rotori compresi).

    All'avvio vengono scritti sul prim `body` del USD: ai rotori resta la loro massa del
    USD e al corpo va il resto (vedi `rigid_body.body_mass_properties`).
    """

    mass_kg: float = 2.5  # massa totale
    cg_m: Vec3 = (0.0, 0.0, 0.0)  # baricentro [x, y, z] nel frame del corpo
    inertia_kg_m2: InertiaConfig = field(default_factory=InertiaConfig)


# ----------------------------------------------------------------------------- propulsione


@dataclass
class RotorLayoutConfig:
    """Posizione e verso di ogni rotore (asse di spinta +z del corpo)."""

    # centro del disco di ogni rotore nel frame del corpo
    positions_m: list[Vec3] = field(default_factory=lambda: [
        (0.18, 0.0, 0.0), (0.0, 0.18, 0.0), (-0.18, 0.0, 0.0), (0.0, -0.18, 0.0)])
    # verso di rotazione visto da +z: "cw" | "ccw"
    directions: list[str] = field(default_factory=lambda: ["cw", "ccw", "cw", "ccw"])

    def __post_init__(self) -> None:
        if len(self.positions_m) != 4 or len(self.directions) != 4:
            raise ValueError("servono esattamente 4 rotori (positions_m e directions)")
        if any(d not in ("cw", "ccw") for d in self.directions):
            raise ValueError(f"directions accetta solo 'cw' e 'ccw', non {self.directions}")
        if self.directions.count("cw") != 2:
            raise ValueError("servono 2 rotori cw e 2 ccw (coppia di imbardata bilanciata)")


@dataclass
class PropellerConfig:
    """Elica, modello di FAST-UAV (PropellerAerodynamicsModel.aero_coefficients_incidence)."""

    diameter_m: float = 0.254
    beta: float = 0.45  # passo/diametro
    n_blades: int = 2
    chord_to_radius: float = 0.15  # corda/raggio a 0,75 R
    # 12 coefficienti del polinomio Ct(beta, J) + J0t; None = default di fastuav
    ct_model: list[float] | None = None
    # 12 coefficienti del polinomio Cp(beta, J) + J0p; None = default di fastuav
    cp_model: list[float] | None = None
    mach_tip_max: float = 0.80  # Mach massimo alla punta (cap sui giri)
    compressibility: bool = True  # penalità di compressibilità sul Cp
    inertia_kg_m2: float = 0.0  # inerzia elica+rotore (non usata)

    def __post_init__(self) -> None:
        for name in ("ct_model", "cp_model"):
            value = getattr(self, name)
            if value is not None and len(value) != 12:
                raise ValueError(f"{name} deve avere 12 coefficienti (10 del polinomio + 2 di J0)")


@dataclass
class MotorConfig:
    """Motore brushless, modello elettrico di FAST-UAV (MotorPerformanceModel)."""

    kv_rpm_per_v: float = 920.0  # costante di velocità Kv in rpm/V
    resistance_ohm: float = 0.10  # resistenza di fase equivalente
    friction_torque_nm: float = 0.01  # coppia d'attrito (da I0/Kv)
    max_torque_nm: float = 0.0  # coppia massima; 0 = nessun limite
    time_constant_s: float = 0.03  # costante di tempo dei giri
    min_rpm: float = 0.0  # giri minimi con il motore armato


@dataclass
class EscConfig:
    """Perdite dell'ESC (ESCPerformanceModel di FAST-UAV); enabled=False -> rendimento 1."""

    enabled: bool = True
    mosfet_resistance_ohm: float = 3.0e-3  # R_ds_on per MOSFET
    diode_voltage_v: float = 0.8  # caduta del diodo di corpo
    dead_time_s: float = 3.0e-7
    pwm_frequency_hz: float = 24000.0
    switching_coefficient: float = 2.0e-8  # perdite di commutazione
    fixed_losses_w: float = 1.0  # perdite fisse (driver, logica, BEC)
    min_efficiency: float = 0.70


@dataclass
class BatteryConfig:
    """Pacco LiPo/Li-ion, curva di scarica di FAST-UAV (physics.battery)."""

    enabled: bool = True  # tensione variabile; False = fissa
    cells_series: int = 4
    capacity_mah: float = 5000.0
    v_full_v: float = 4.20  # tensione di cella carica
    v_mean_v: float = 3.70  # tensione nominale di cella
    v_cut_v: float = 3.30  # tensione di cella a fine scarica
    resistance_ohm: float = 0.02  # resistenza interna del pacco
    initial_dod: float = 0.0  # scarica iniziale (0 = piena)
    max_dod: float = 0.95  # scarica massima utilizzabile

    def __post_init__(self) -> None:
        if not self.v_cut_v < self.v_mean_v < self.v_full_v:
            raise ValueError("serve v_cut_v < v_mean_v < v_full_v")


# ----------------------------------------------------------------------------- aerodinamica


@dataclass
class BodyAeroConfig:
    """Fusoliera come corpo di rivoluzione con asse lungo +z (Jorgensen, MultirotorAero)."""

    cd0: float = 0.10  # coefficiente assiale CD0 riferito a S_front
    s_front_m2: float = 0.010  # area frontale (vista lungo l'asse)
    s_top_m2: float = 0.030  # area in pianta (vista di traverso)
    eta: float = 0.65  # fattore di lunghezza finita (Hoerner)
    cd_crossflow: float = 1.2  # Cd trasversale del cilindro
    cl0: float = 0.0  # C_L0 del velivolo


@dataclass
class WingConfig:
    """Bracci-ala del tailsitter (arm_wing_forces di FAST-UAV); disattivati di default."""

    enabled: bool = False
    area_single_m2: float = 0.02  # area di un braccio-ala
    aspect_ratio: float = 2.5
    cl_max: float = 1.35
    cd0_wing: float = 0.012  # Cd0 di profilo
    oswald: float = 0.85
    n_lift: float = 2.0  # superfici che portano
    n_drag: float = 4.0  # superfici che danno resistenza di profilo


@dataclass
class SpinnerConfig:
    enabled: bool = False  # ogive in punta ai bracci
    count: int = 4
    diameter_m: float = 0.03
    cd: float = 0.05


@dataclass
class StationsConfig:
    """Ascisse dal MUSO (positive verso la coda) per il momento di beccheggio aerodinamico."""

    enabled: bool = False
    nose_z_m: float = 0.30  # quota z del muso nel frame del corpo
    body_length_m: float = 0.60
    body_volume_ratio: float = 0.6
    body_planform_centroid_m: float = 0.30  # centroide della pianta dal muso
    wing_m: float = 0.40  # bracci-ala dal muso
    spinner_m: float = 0.30  # spinner dal muso


@dataclass
class AeroConfig:
    """Aerodinamica dell'airframe: MultirotorAero di FAST-UAV, generalizzato in 3D."""

    enabled: bool = True
    body: BodyAeroConfig = field(default_factory=BodyAeroConfig)
    wing: WingConfig = field(default_factory=WingConfig)
    spinner: SpinnerConfig = field(default_factory=SpinnerConfig)
    stations: StationsConfig = field(default_factory=StationsConfig)
    min_airspeed_m_s: float = 0.3  # sotto: forze nulle


@dataclass
class AtmosphereConfig:
    rho_kg_m3: float = 1.225
    speed_of_sound_m_s: float = 340.3
    kinematic_viscosity_m2_s: float = 1.46e-5
    wind_m_s: Vec3 = (0.0, 0.0, 0.0)  # vento costante [x, y, z] ENU


@dataclass
class PhysicsLogConfig:
    enabled: bool = False  # scrive un CSV della fisica
    path: str = "logs/physics.csv"
    period_s: float = 0.01  # periodo di campionamento


# ----------------------------------------------------------------------------- insieme


@dataclass
class PhysicsConfig:
    """Tutto il modello: corpo rigido, rotori, elica, motore, ESC, batteria, aero, atmosfera."""

    rigid_body: RigidBodyConfig = field(default_factory=RigidBodyConfig)
    rotors: RotorLayoutConfig = field(default_factory=RotorLayoutConfig)
    propeller: PropellerConfig = field(default_factory=PropellerConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    esc: EscConfig = field(default_factory=EscConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    aero: AeroConfig = field(default_factory=AeroConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    log: PhysicsLogConfig = field(default_factory=PhysicsLogConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PhysicsConfig:
        """Dict anche parziale (le chiavi mancanti prendono il default); chiavi sconosciute: errore."""
        return from_dict(cls, data)

    @classmethod
    def coerce(cls, value: PhysicsConfig | Mapping[str, Any]) -> PhysicsConfig:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- da dict

T = typing.TypeVar("T")


def from_dict(cls: type[T], data: Mapping[str, Any] | None) -> T:
    """Costruisce la dataclass `cls` (e quelle annidate) da un dict, anche parziale."""
    if data is None:
        return cls()
    if isinstance(data, cls):
        return data
    if not isinstance(data, Mapping):
        raise TypeError(f"{cls.__name__}: atteso un dict, ricevuto {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{cls.__name__}: chiavi sconosciute {unknown} (valide: {sorted(names)})")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if is_dataclass(hint):
            value = from_dict(hint, value)
        elif isinstance(value, tuple):
            value = list(value) if typing.get_origin(hint) is list else value
        kwargs[name] = value
    return cls(**kwargs)

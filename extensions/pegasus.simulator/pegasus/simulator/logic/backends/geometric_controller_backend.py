"""
| File: geometric_controller_backend.py
| Description: Backend con il controllore di prova (geometrico su SO(3)) tarato sulla fisica
|              avanzata del Multirotor (MultirotorConfig.physics). Serve a far volare il drone
|              senza autopilota esterno mentre si prova la fisica.
| License: BSD-3-Clause.

Pegasus chiama `input_reference()` all'inizio di Multirotor.update e `update(dt)` alla fine:
il comando calcolato con lo stato di un passo si applica al passo successivo, come con un
autopilota vero.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig
from pegasus.simulator.logic.physics.config import PhysicsConfig
from pegasus.simulator.logic.physics.controller import GeometricControllerConfig, build_controller
from pegasus.simulator.logic.physics.frames import BodyState
from pegasus.simulator.logic.physics.model import AdvancedMultirotorPhysics


class GeometricControllerBackend(Backend):
    """Comanda i giri dei rotori per tenere un punto, seguire waypoint o fare la transizione.

    Args:
        controller: GeometricControllerConfig o dict (modalità, setpoint, guadagni).
        physics: la stessa PhysicsConfig (o dict) passata a MultirotorConfig.physics: il
            controllore ne usa massa, inerzia, bracci e costanti statiche dell'elica.
    """

    def __init__(self, controller: GeometricControllerConfig | Mapping[str, Any] | None = None,
                 physics: PhysicsConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__(BackendConfig())
        self.controller_config = GeometricControllerConfig.coerce(controller)
        model = AdvancedMultirotorPhysics(PhysicsConfig.coerce(physics or {}))
        self.controller = build_controller(self.controller_config, model)
        self._state: Any = None
        self._command = [0.0] * 4

    def update_sensor(self, sensor_type: str, data: Any) -> None:
        pass

    def update_graphical_sensor(self, sensor_type: str, data: Any) -> None:
        pass

    def update_state(self, state: Any) -> None:
        self._state = state

    def input_reference(self) -> list[float]:
        return self._command

    def update(self, dt: float) -> None:
        if self._state is None:
            return
        state = BodyState.from_pegasus(self._state)
        self._command = [float(w) for w in self.controller.update(state, dt)]

    def start(self) -> None:
        self.reset()

    def stop(self) -> None:
        self._command = [0.0] * 4

    def reset(self) -> None:
        self.controller.reset()
        self._command = [0.0] * 4

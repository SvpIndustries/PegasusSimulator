"""CSV della fisica (PhysicsConfig.log): stesso formato in Isaac e offline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import IO, Any

import numpy as np

from pegasus.simulator.logic.physics.frames import BodyState
from pegasus.simulator.logic.physics.model import Wrench
from pegasus.simulator.logic.physics.config import PhysicsLogConfig


class PhysicsLogger:
    def __init__(self, settings: PhysicsLogConfig) -> None:
        self.s = settings
        self._file: IO[str] | None = None
        self._writer: Any = None
        self._next_t = 0.0

    def write(self, t: float, state: BodyState, wrench: Wrench) -> None:
        if not self.s.enabled or t < self._next_t:
            return
        self._next_t = t + self.s.period_s
        r = state.rotation
        row: dict[str, float] = {"t": t}
        row.update({f"p{a}": float(v) for a, v in zip("xyz", state.position, strict=True)})
        row.update({f"v{a}": float(v) for a, v in zip("xyz", state.linear_velocity, strict=True)})
        row.update({f"w{a}": float(v) for a, v in zip("xyz", state.angular_velocity, strict=True)})
        row["tilt_deg"] = float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))
        row.update(wrench.telemetry)
        if self._writer is None:
            path = Path(self.s.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

"""Da dove viene fastuav: percorso, commit e modifiche locali, per i log di ogni prova."""

from __future__ import annotations

import subprocess
from pathlib import Path


def fastuav_version() -> str:
    import fastuav

    package = Path(fastuav.__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(package), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(package), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return f"{package} (non in un repo git)"
    return f"{package} @ {commit}{' + modifiche locali' if dirty else ''}"

"""
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
"""

# Isaac Sim / Kit modules are only available inside the simulator. Outside of it (offline
# physics, tests, configuration checks) the pure-Python parts of the package stay importable.
try:
    from .interface.pegasus_interface import PegasusInterface
except ImportError as _error:
    if (_error.name or "").split(".")[0] not in ("omni", "carb", "pxr", "isaacsim", "usdrt"):
        raise

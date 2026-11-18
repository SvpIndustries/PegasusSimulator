"""Ambiente dei processi SITL: niente percorsi Python/librerie di Isaac Sim."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Caricato dal file: il pacchetto backends importa Isaac (carb) all'import.
_PATH = (
    Path(__file__).resolve().parents[2]
    / "pegasus/simulator/logic/backends/tools/ardupilot_launch_tool.py"
)
pytest.importorskip("psutil")
_spec = importlib.util.spec_from_file_location("ardupilot_launch_tool", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

ISAAC = "/opt/isaac-test"


def _env(**extra):
    return {"ISAAC_PATH": ISAAC, "HOME": "/home/u", "PATH": "/usr/bin", **extra}


def test_drops_isaac_python_and_preload():
    env = tool.clean_sitl_environment(
        _env(PYTHONHOME=f"{ISAAC}/kit/python", LD_PRELOAD=f"{ISAAC}/kit/libcarb.so")
    )
    assert "PYTHONHOME" not in env
    assert "LD_PRELOAD" not in env


def test_filters_isaac_entries_and_keeps_others():
    env = tool.clean_sitl_environment(
        _env(
            PYTHONPATH=os.pathsep.join(
                [f"{ISAAC}/kit/python/lib/python3.11", "/opt/ros/jazzy/lib/python3.12/site-packages"]
            ),
            LD_LIBRARY_PATH=os.pathsep.join([f"{ISAAC}/.", f"{ISAAC}/exts/x/lib", "/opt/ros/jazzy/lib"]),
        )
    )
    assert env["PYTHONPATH"] == "/opt/ros/jazzy/lib/python3.12/site-packages"
    assert env["LD_LIBRARY_PATH"] == "/opt/ros/jazzy/lib"


def test_removes_lists_left_empty():
    env = tool.clean_sitl_environment(
        _env(PYTHONPATH=f"{ISAAC}/kit/python/lib/python3.11", LD_LIBRARY_PATH=f"{ISAAC}/.")
    )
    assert "PYTHONPATH" not in env
    assert "LD_LIBRARY_PATH" not in env


def test_prefix_is_not_a_false_match():
    env = tool.clean_sitl_environment(_env(PYTHONPATH=f"{ISAAC}-other/lib"))
    assert env["PYTHONPATH"] == f"{ISAAC}-other/lib"


def test_running_interpreter_counts_as_isaac():
    env = tool.clean_sitl_environment(
        {"PATH": "/usr/bin", "PYTHONPATH": f"{sys.prefix}/lib/python3.11"}
    )
    assert "PYTHONPATH" not in env


def test_does_not_touch_the_source_environment():
    source = _env(PYTHONHOME=f"{ISAAC}/kit/python", PATH="/usr/bin")
    tool.clean_sitl_environment(source)
    assert source["PYTHONHOME"] == f"{ISAAC}/kit/python"
    assert tool.clean_sitl_environment(source)["PATH"] == "/usr/bin"


def test_sitl_binary_per_vehicle():
    assert tool.sitl_binary("ArduCopter") == "arducopter"
    assert tool.sitl_binary("ArduPlane") == "arduplane"
    assert tool.sitl_binary("Blimp") == "blimp"


def test_default_vehicle_is_copter_and_binary_check_follows_the_vehicle(tmp_path):
    (tmp_path / "build/sitl/bin").mkdir(parents=True)
    (tmp_path / "build/sitl/bin/arduplane").write_text("")
    copter = tool.ArduPilotLaunchTool(str(tmp_path))
    plane = tool.ArduPilotLaunchTool(str(tmp_path), vehicle="ArduPlane")
    assert copter.vehicle == "ArduCopter" and not copter._sitl_already_exists()
    assert plane._sitl_already_exists()

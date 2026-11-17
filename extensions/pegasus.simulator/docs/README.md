# Pegasus Simulator

Pegasus Simulator is a framework built on top of NVIDIA
Omniverse and Isaac
Sim. It is designed to provide an easy, yet powerfull way of simulating the dynamics of vehicles. It provides a simulation interface for PX4 integration as well as a custom python control interface. At the moment, only multirotor vehicles are supported, with support for other vehicle topologies planned for future versions.

## Advanced multirotor physics (SvpIndustries fork)

`Multirotor` keeps the original model by default (quadratic thrust curve, linear drag).
Setting `MultirotorConfig.physics` enables the advanced model in
`pegasus/simulator/logic/physics/`, built on `fastuav.physics` (FAST-UAV):

- motors: first-order speed response, limited by battery voltage (`Kv (U - R I)`) and torque;
  electrical model of motor and ESC;
- propellers: `Ct`, `Cp` as functions of pitch, advance ratio and disk incidence, tip-Mach cap;
- battery: open-circuit voltage vs. depth of discharge, sag under load;
- airframe: body, arm-wings and spinners (`MultirotorAero`), extended to 3D, with constant wind;
- rigid body: mass, CG and inertia of the complete drone written to the USD `body` prim at spawn.

```python
from pegasus.simulator.logic.backends import GeometricControllerBackend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

physics = {"rigid_body": {"mass_kg": 5.97}, "motor": {"kv_rpm_per_v": 1040.0},
           "battery": {"cells_series": 8}}             # partial dict: the rest are defaults
config = MultirotorConfig()
config.physics = physics                               # or a PhysicsConfig
config.backends = [GeometricControllerBackend({"setpoint_m": (0.0, 0.0, 2.0)}, physics)]
Multirotor("/World/quadrotor", usd_path, 0, [0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0], config=config)
```

All parameters, units and defaults: `pegasus/simulator/logic/physics/config.py`. In YAML vehicle
configs the same dict goes under the `physics` key. Conventions: body FLU, rotor thrust along
+z, world ENU. The physics package is pure Python: `logic/physics/bench.py` runs it offline
(hover, motor step, level-flight trim, tailsitter transition, bench comparison).

Requirements: `fastuav` is not on PyPI; install a local FAST-UAV copy in the Isaac Sim Python
with `python.sh -m pip install --no-deps -e <FAST-UAV>`. Without it the default model works
as before. Tests (no Isaac Sim needed):

```bash
<ISAACSIM_PATH>/python.sh -m pytest extensions/pegasus.simulator/tests/physics
```

## Contributing

The developers of the Pegasus simulator welcome any positive contributions and ideas from the robotics comunity to make
in order to allow this extension to mature. If you think you have a nice contribution to make or just a simple suggestion,
feel free to create bug reports, feature requests or open pull requests for direct code contributions.

## Acknowledgement

NVIDIA Isaac Sim is available freely under https://www.nvidia.com/en-us/omniverse/download/.

Pegasus Simulator is released under BSD-3 License.
The license files of its dependencies and assets are present in the docs/licenses directory.

## Citation
Please cite if you use this extension in your work:

```
@misc{jacinto2023pegasus,
	author = {Marcelo Jacinto and Rita Cunha},
	title = {Pegasus Simulator: An Isaac Sim Framework for Multiple Aerial Vehicles Simulation},
	year = {2023},
	eprint = {},
}
```

## Main Developer Team

This simulation framework is an open-source effort, started by me, Marcelo Jacinto in January/2023. It is a tool that was created with the original purpose of serving my Ph.D. workplan for the next 4 years, which means that you can expect this repository to be mantained, hopefully at least until 2027.

* Project Founder
	* Marcelo Jacinto], under the supervision of Prof. Rita Cunha and Prof. Antonio Pascoal (IST/ISR-Lisbon)
* Architecture
    * Marcelo Jacinto
	* João Pinto
* Multirotor Dynamic Simulation and Control
    * Marcelo Jacinto
* Example Applications
	* Marcelo Jacinto
	* João Pinto
* Paper Writting and Revision
	* Marcelo Jacinto
	* João Pinto
	* Rita Cunha
	* António Pascoal
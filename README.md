# RotorPy
A Python-based multirotor simulation environment with aerodynamic wrenches, useful for education and research in estimation, planning, and control for UAVs.
<p align="center"><img src="/media/double_pillar.gif" width="32%"/><img src="/media/gusty.gif" width="32%"/><img src="/media/minsnap.gif" width="32%"/></p>

**NEW in `v3.0`**: RotorPy now includes a fast (non-photorealistic) camera sensor for developing and testing vision algorithms (visual odometry, vision-based policies, etc.). Check out the new examples: [`camera_demo.py`](/examples/camera_demo.py), [`camera_visualization.py`](/examples/camera_visualization.png), and  [`visual_servoing_policy.py`](/examples/visual_servoing_policy.py).
<p align="center"><img src="/media/camera_demo.gif" width="50%"/><img src="/media/visual_servoing_example.gif" width="50%">
<p align="center"><img src="/media/camera_highlight.png" width="100%">

World blocks can now carry visual features: 3D points with RGB colors and optional descriptor vectors. The new `PinholeCamera` sensor projects those features into an image using the standard pinhole + distortion model. `BatchedPinholeCamera` renders images for many UAVs in parallel. 

## Purpose and Model Scope
The original focus of this simulator was on accurately simulating rotary-wing UAV dynamics using lumped parameter representations of the aerodynamics, primarily for graduate level course design and exploratory research. These aerodynamic effects, outlined in [`rotorpy/vehicles/README.md`](/rotorpy/vehicles/README.md), are negligible at hover in still air; however, as relative airspeed increases (e.g. for aggressive maneuvers or in the presence of high winds), they quickly become noticeable and force the student/researcher to reconcile with them. 

As RotorPy continues to grow, the focus is now on building a sufficiently realistic dynamics simulator that can also scale to quickly generate extremely large datasets as a tool for deep learning, reinforcement learning, and Monte Carlo studies on existing (or new!) algorithms in estimation, planning, and control. 

The engine is designed from the bottom up to be lightweight, easy to install with very limited dependencies or requirements, and interpretable to anyone with basic working knowledge of Python. The intent is for users to gain intuition about UAV dynamics and learn how to develop control and/or estimation algorithms for rotary wing vehicles especially in the presence of aerodynamic wrenches and strong winds. 

We hope that this repository will be a helpful resource for educators and researchers. 

# Installation

RotorPy can be installed using `pip`:

```bash
pip install rotorpy 
```

This will install the minimum packages to run the most basic version of the simulator. But to get the most of RotorPy you may want to consider...

```bash
pip install rotorpy.[all]       # the complete package, includes everything below too
pip install rotorpy.[testing]   # if you're planning on developing and need to run the testing suite
pip install rotorpy.[batched]   # for the batched (parallelized) environments
pip install rotorpy.[learning]  # for learning with the gymnasium environment
pip install rotorpy.[px4]       # for integration with PX4
```

For other tagged versions, see `pyproject.toml`. It's an evolving thing.

# Usage

There are a few example scripts found in `rotorpy/examples/` that demonstrate how to use RotorPy in a variety of ways including for Monte Carlo evaluations, reinforcement learning, swarms, and even visual servoing (as of `v3.0`). 

#### Regular usage
A good place to start would be to reference the `rotorpy/examples/basic_usage.py` script. It goes through the necessary imports and how to create and execute an instance of the base simulation environment. 
 
At minimum the simulator requires vehicle, controller, and trajectory objects. The vehicle (and potentially the controller) is parameterized by a unique parameter file, such as `rotorpy/vehicles/hummingbird_params.py`. There is also the option to specify your own IMU, world bounds, and how long you would like to run the simulator for. 

The output of the simulator is a dictionary containing a time vector and the time histories of all the vehicle's states, inputs, and measurements.

Below is a minimum working example: 

```python
import numpy as np
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.wind.default_winds import SinusoidWind

sim_instance = Environment(vehicle=Multirotor(quad_params),           # vehicle object, must be specified. 
                           controller=SE3Control(quad_params),        # controller object, must be specified.
                           trajectory=TwoDLissajous(),                # trajectory object, must be specified.
                           wind_profile=SinusoidWind(),               # OPTIONAL: wind profile object, if none is supplied it will choose no wind. 
                           sim_rate     = 100,                        # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                           camera       = None,                       # OPTIONAL: camera sensor object, by default won't use the camera if none is supplied.
                           imu          = None,                       # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                           mocap        = None,                       # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.  
                           estimator    = None,                       # OPTIONAL: estimator object
                           world        = None,                       # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                           safety_margin= 0.25                        # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                       )

x0 = {'x': np.array([0,0,0]),
      'v': np.zeros(3,),
      'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
      'w': np.zeros(3,),
      'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
      'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
sim_instance.vehicle.initial_state = x0

# The results are a dictionary containing the relevant state, input, and measurements vs time.
results = sim_instance.run(t_final      = 20,       # The maximum duration of the environment in seconds
                           use_mocap    = False,       # Boolean: determines if the controller should use the motion capture estimates. 
                           terminate    = False,       # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                           plot            = True,     # Boolean: plots the vehicle states and commands   
                           plot_mocap      = True,     # Boolean: plots the motion capture pose and twist measurements
                           plot_estimator  = True,     # Boolean: plots the estimator filter states and covariance diagonal elements
                           plot_imu        = True,     # Boolean: plots the IMU measurements
                           animate_bool    = True,     # Boolean: determines if the animation of vehicle state will play. 
                           animate_wind    = True,    # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV. 
                           verbose         = True,     # Boolean: will print statistics regarding the simulation. 
                           fname   = None # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/. 
                    )

```
#### Batched Simulations
RotorPy includes a batched environment which can simulate multiple drones in parallel on CPU or GPU. For simulations of >1000 drones, we have observed speedups of 25x purely on CPU compared to simulating all drones sequentially. We have also implemented batched versions of existing control, trajectory, sensor, and wind classes. See [`examples/batched_simulation.py`](/examples/batched_simulation.py) for how to use the batched simulation, and [`examples/benchmark_batched_simulation.py`](/examples/benchmark_batched_simulation.py) to measure the speedup on your own system.

#### Reinforcement Learning
RotorPy includes a custom Gymnasium environment, `QuadrotorEnv`, which is a stripped down version of the regular simulation environment intended for applications in reinforcement learning. `QuadrotorEnv` features all the aerodynamics and motor dynamics, but also supports different control abstractions ranging from high level velocity vector commands all the way down to direct individual motor speed commands. This environment also allows the user to specify their own reward function. 

For an example of how to interface with this environment, see `rotorpy/examples/gymnasium_basic_usage.py`. You can also see an example of training a quadrotor to hover using this environment in `rotorpy/examples/ppo_hover_train.py` and `rotorpy/examples/ppo_hover_eval.py`. 

You can find this new environment in the `rotorpy/learning/` module. 

#### And much more
RotorPy is intended to be flexible to support the aerial robotics field as it expands. Explore the examples in this repository and [citations](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17472125763442225756) to see the many ways RotorPy can be used to learn more about aerial robotics. 

# Development

It is rather straightforward if you would like to add more tracking methods into the simulator. For instance, if you'd like to add a new trajectory generator or a new controller, we've added respective templates that you can use under `rotorpy/trajectories/` and `rotorpy/controllers/` to help structure your code appropriately. If you'd like to add your own wind field, you can add a new class in `rotorpy/wind/` following the template there. 

As for adding more core functionality (e.g., sensors, new vehicle dynamics, animations, etc.), those require a bit more effort to make sure that all the inputs and outputs are set up accordingly. One piece of advice is that the main loop occurs in `rotorpy/simulate.py`. Under the `while` loop, you can see the process by which the vehicle dynamics, trajectory generator, IMU sensor, and controller interface with each other. 


# Citation

If you use RotorPy for your work please cite our companion workshop paper contributed to the [RS4UAVs Workshop at ICRA 2023](https://imrclab.github.io/workshop-uav-sims-icra2023/): 

```
@article{folk2023rotorpy,
  title={{RotorPy}: A Python-based Multirotor Simulator with Aerodynamics for Education and Research},
  author={Folk, Spencer and Paulos, James and Kumar, Vijay},
  journal={arXiv preprint arXiv:2306.04485},
  year={2023}
}
```

See how academics have been using RotorPy: 
[![Citations](https://img.shields.io/badge/Citations-Google%20Scholar-green?logo=googlescholar)](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17472125763442225756)

# Acknowledgements

We would like to acknowledge [Jimmy Paulos](https://github.com/jpaulos) who wrote the majority of the underlying engine for RotorPy, and the teaching assistants who contributed code to the initial version of this simulator, especially [Dan Mox](https://github.com/danmox), [Laura Jarin-Lipschitz](https://github.com/ljarin), [Rebecca Li](https://github.com/rebeccali), [Shane Rozen-Levy](https://github.com/ShaneRozenLevy), [Xu Liu](https://github.com/XuRobotics), [Yuezhan Tao](https://github.com/tyuezhan), [Yu-Ming Chen](https://github.com/yminchen), and [Fernando Cladera](https://github.com/fcladera). 

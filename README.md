# RotorPy
A Python-based multirotor simulation environment with aerodynamic wrenches, useful for education and research in estimation, planning, and control for UAVs.
<p align="center"><img src="/media/double_pillar.gif" width="32%"/><img src="/media/gusty.gif" width="32%"/><img src="/media/minsnap.gif" width="32%"/></p>

**NEW in `v3.0`**: RotorPy now includes a fast, non-photorealistic camera sensor for developing and testing vision algorithms (visual odometry, image correspondence, learning-based vision policies, etc.). World blocks can now carry visual features (3D points with RGB colors and optional descriptor vectors) specified either directly in the world JSON or using generator scripts.

`PinholeCamera.render()` projects those features into an image with a customizable pinhole + distortion model and explicit occlusion handling, and `BatchedPinholeCamera` renders for many drones in parallel on CPU or GPU. Each output pairs per-feature 3D/keypoint information with the RGB `colors` and descriptor `descriptors`, and the `feature_output` toggle (`'all'`/`'rgb'`/`'descriptors'`) trims either when collecting large datasets. See `rotorpy/examples/camera_visualization.py` for a complete example.

<!-- TODO: Add images or animations highlighting the new camera capability. -->

## Purpose and Model Scope
The original focus of this simulator was on accurately simulating rotary-wing UAV dynamics with added lumped parameter representations of the aerodynamics for course design and exploratory research. These aerodynamic effects, listed below, are negligible at hover in still air; however, as relative airspeed increases (e.g. for aggressive maneuvers or in the presence of high winds), they quickly become noticeable and force the student/researcher to reconcile with them. 

As RotorPy continues to grow, the focus is now on building a sufficiently realistic dynamics simulator that can scale to quickly generate thousands (or even millions) of simulated rotary-wing UAVs as a tool for deep learning, reinforcement learning, and Monte Carlo studies on existing (or new!) algorithms in estimation, planning, and control. 

The engine is designed from the bottom up to be lightweight, easy to install with very limited dependencies or requirements, and interpretable to anyone with basic working knowledge of Python. The intent is for users to gain intuition about UAV dynamics/aerodynamics and learn how to develop control and/or estimation algorithms for rotary wing vehicles especially in the presence of aerodynamic wrenches. 

The following aerodynamic effects of interest are within the scope of this model: 
1. **Parasitic Drag** - Drag associated with non-lifting surfaces like the frame. This drag is quadratic in airspeed. 
2. **Rotor Drag** - This is an apparent drag force that is a result of the increased drag produced by the advancing blade of a rotor. Rotor drag is linear in airspeed. 
3. **Blade Flapping** - An effect of dissymmetry of lift, blade flapping is the motion of the blade up or down that results in a pitching moment. The pitching moment is linear in the airspeed. 
4. **Induced Drag** - Another effect of dissymmetry of lift, more apparent in semi-rigid or rigid blades, where an increase of lift on the advancing blade causes an increased induced downwash, which in turn tilts the lift vector aft resulting in more drag. Induced drag is linear in the airspeed. 
5. **Translational Lift** - In forward motion, the induced velocity at the rotor plane decreases, causing an increase in lift generation. 
6. **Translational Drag** - A consequence of translational lift, and similar to **Induced Drag**, the increased lift produced in forward flight will produce an increase in induced drag on the rotor. 

Ultimately the effects boil down to forces acting anti-parallel to the relative airspeed and a combination of pitching moments acting parallel and perpendicular to the relative airspeed. The rotor aerodynamic effects (rotor drag, blade flapping, induced drag, and translational drag) can be lumped into a single drag force acting at each rotor hub, whereas parasitic drag can be lumped into a single force and moment vector acting at the center of mass. 

What's currently ignored: any lift produced by the frame or any torques produced by an imbalance of drag forces on the frame. We also currently neglect variations in the wind along the length of the UAV, implicitly assuming that the characteristic length scales of the wind fields are larger than UAV's maximum dimensions. Remember, the drag models used here are representative of bluff bodies, not wings. 

RotorPy also includes first-order motor dynamics to simulate lag, as well as support for spatio-temporal wind flow fields for the UAV to interact with. 

# Installation

RotorPy can be installed using `pip`:

```
pip install rotorpy 
```

To install learning dependencies, run: 

```
pip install rotorpy.[learning]
```

In addition to the base requirements, it will install `stable_baselines3` and `tensorboard`. For other tagged versions, see `pyproject.toml`. 

# Usage

There are a few example scripts found in `rotorpy/examples/` that demonstrate how to use RotorPy in a variety of ways including for Monte Carlo evaluations, reinforcement learning, and swarms. 

#### Regular usage
A good place to start would be to reference the `rotorpy/examples/basic_usage.py` script. It goes through the necessary imports and how to create and execute an instance of the simulator. 
 
At minimum the simulator requires vehicle, controller, and trajectory objects. The vehicle (and potentially the controller) is parameterized by a unique parameter file, such as in `rotorpy/vehicles/hummingbird_params.py`. There is also the option to specify your own IMU, world bounds, and how long you would like to run the simulator for. 

The output of the simulator is a dictionary containing a time vector and the time histories of all the vehicle's states, inputs, and measurements.

Below is a minimum working example: 

```
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
                           imu          = None,                       # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                           mocap        = None,                       # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.  
                           estimator    = None,                       # OPTIONAL: estimator object
                           world        = None,                      # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
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

#### Vision
New in `v3.0.0`! RotorPy includes a fast, non-photorealistic pinhole camera sensor for learning or developing computer vision algorithms (visual odometry, image correspondence, etc.). World blocks can carry visual features (3D positions with RGB colors and optional descriptor vectors, e.g. 128-d SIFT) placed by pluggable generators (`World.grid_forest(..., add_features=True)`), supporting regular surface grids, areal-density random splatter, uniform or random placement along block edges, and synthetic `descriptor_dim`-dimensional feature vectors. Features can also be embedded directly in the world JSON for exact reproducibility across trials, without relying on random seeds. `PinholeCamera.render()` projects those features into an image with a customizable pinhole + distortion model while explicitly handling occlusion too. `render()`/`measurement()` outputs pair per-feature 3D/keypoint information with both the RGB `colors` and the `descriptors`, so the same ground-truth can feed descriptor-based VO/VIO matching, dense depth, or vision policies; pass `feature_output` (`'all'`/`'rgb'`/`'descriptors'`) to return both or to trim either the colors or the descriptor arrays when collecting large datasets. A torch-parallel `BatchedPinholeCamera` renders from many vehicles simultaneously, with per-drone domain randomization — every drone can carry its own intrinsics, extrinsics, near plane, splat radius, and noise profile (sampled by `randomize_camera_params()`), and can even observe its own `World` via the per-drone-worlds path — for scalable sim-to-real training pipelines. See `rotorpy/examples/camera_visualization.py` for a complete example, and `rotorpy/sensors/` for the sensor docs.

#### Reinforcement Learning
New in `v1.1.0`, RotorPy includes a custom Gymnasium environment, `QuadrotorEnv`, which is a stripped down version of the regular simulation environment intended for applications in reinforcement learning. `QuadrotorEnv` features all the aerodynamics and motor dynamics, but also supports different control abstractions ranging from high level velocity vector commands all the way down to direct individual motor speed commands. This environment also allows the user to specify their own reward function. 

For an example of how to interface with this environment, see `rotorpy/examples/gymnasium_basic_usage.py`. You can also see an example of training a quadrotor to hover using this environment in `rotorpy/examples/ppo_hover_train.py` and `rotorpy/examples/ppo_hover_eval.py`. 

You can find this new environment in the `rotorpy/learning/` module. 

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

[![Citations](https://img.shields.io/badge/Citations-Google%20Scholar-green?logo=googlescholar)](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17472125763442225756)

# Acknowledgements

We would like to acknowledge [Jimmy Paulos](https://github.com/jpaulos) who wrote the majority of the underlying engine for RotorPy, and the teaching assistants who contributed code to the initial version of this simulator, especially [Dan Mox](https://github.com/danmox), [Laura Jarin-Lipschitz](https://github.com/ljarin), [Rebecca Li](https://github.com/rebeccali), [Shane Rozen-Levy](https://github.com/ShaneRozenLevy), [Xu Liu](https://github.com/XuRobotics), [Yuezhan Tao](https://github.com/tyuezhan), [Yu-Ming Chen](https://github.com/yminchen), and [Fernando Cladera](https://github.com/fcladera). 

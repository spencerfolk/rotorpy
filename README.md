# RotorPy
A Python-based multirotor simulation environment with aerodynamic wrenches, useful for education and research in estimation, planning, and control for UAVs.
<p align="center"><img src="/media/double_pillar.gif" width="32%"/><img src="/media/gusty.gif" width="32%"/><img src="/media/minsnap.gif" width="32%"/></p>

**NEW in `v1.1.0`**: RotorPy now includes a customizable [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment found in the new `rotorpy/learning/` module. 

<p align="center"><img src="/media/ppo_hover_20k.gif" width="32%"/><img src="/media/ppo_hover_600k.gif" width="32%"/><img src="/media/ppo_hover_1000k.gif" width="32%"/></p>

## Purpose and Scope
The original focus of this simulator was on accurately simulating rotary-wing UAV dynamics with added lumped parameter representations of the aerodynamics for course design and exploratory research. These aerodynamic effects, listed below, are negligible at hover in still air; however, as relative airspeed increases (e.g. for aggressive maneuvers or in the presence of high winds), they quickly become noticeable and force the student/researcher to reconcile with them. 

As RotorPy continues to grow, the focus is now on building a realistic dynamics simulator that can scale to quickly generate thousands (or even millions) of simulated rotary-wing UAVs for applications in deep learning, reinforcement learning, and Monte Carlo studies on existing (or new!) algorithms in estimation, planning, and control. 

The engine is designed from the bottom up to be lightweight, easy to install with very limited dependencies or requirements, and interpretable to anyone with basic working knowledge of Python. The simulator is intended to gain intuition about UAV dynamics/aerodynamics and learn how to develop control and/or estimation algorithms for rotary wing vehicles subject to aerodynamic wrenches. 

The following aerodynamic effects of interest are within the scope of this model: 
1. **Parasitic Drag** - Drag associated with non-lifting surfaces like the frame. This drag is quadratic in airspeed. 
2. **Rotor Drag** - This is an apparent drag force that is a result of the increased drag produced by the advancing blade of a rotor. Rotor drag is linear in airspeed. 
3. **Blade Flapping** - An effect of dissymmetry of lift, blade flapping is the motion of the blade up or down that results in a pitching moment. The pitching moment is linear in the airspeed. 
4. **Induced Drag** - Another effect of dissymmetry of lift, more apparent in semi-rigid or rigid blades, where an increase of lift on the advancing blade causes an increased induced downwash, which in turn tilts the lift vector aft resulting in more drag. Induced drag is linear in the airspeed. 
5. **Translational Lift** - In forward motion, the induced velocity at the rotor plane decreases, causing an increase in lift generation. 
6. **Translational Drag** - A consequence of translational lift, and similar to **Induced Drag**, the increased lift produced in forward flight will produce an increase in induced drag on the rotor. 

Ultimately the effects boil down to forces acting anti-parallel to the relative airspeed and a combination of pitching moments acting parallel and perpendicular to the relative airspeed. The rotor aerodynamic effects (rotor drag, blade flapping, induced drag, and translational drag) can be lumped into a single drag force acting at each rotor hub, whereas parasitic drag can be lumped into a single force and moment vector acting at the center of mass. 

What's currently ignored: any lift produced by the frame or any torques produced by an imbalance of drag forces on the frame. We also currently neglect variations in the wind along the length of the UAV, implicitly assuming that the characteristic length scales of the wind fields are larger than UAV's maximum dimensions.

RotorPy also includes first-order motor dynamics to simulate lag, as well as support for spatio-temporal wind flow fields for the UAV to interact with. 

# Installation

RotorPy can be installed using `pip`:

```
pip install rotorpy 
```

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
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.wind.default_winds import SinusoidWind

sim_instance = Environment(vehicle=Multirotor(quad_params),           # vehicle object, must be specified. 
                           controller=SE3Control(quad_params),        # controller object, must be specified.
                           trajectory=CircularTraj(radius=2),         # trajectory object, must be specified.
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

#### Reinforcement Learning
New in `v1.1.0`, RotorPy includes a custom Gymnasium environment, `QuadrotorEnv`, which is a stripped down version of the regular simulation environment intended for applications in reinforcement learning. `QuadrotorEnv` features all the aerodynamics and motor dynamics, but also supports different control abstractions ranging from high level velocity vector commands all the way down to direct individual motor speed commands. This environment also allows the user to specify their own reward function. 

For an example of how to interface with this environment, see `rotorpy/examples/gymnasium_basic_usage.py`. You can also see an example of training a quadrotor to hover using this environment in `rotorpy/examples/ppo_hover_train.py` and `rotorpy/examples/ppo_hover_eval.py`. 

You can find this new environment in the `rotorpy/learning/` module. 

# Development

It is rather straightforward if you would like to add more tracking methods into the simulator. For instance, if you'd like to add a new trajectory generator or a new controller, we've added respective templates that you can use under `rotorpy/trajectories/` and `rotorpy/controllers/` to help structure your code appropriately. If you'd like to add your own wind field, you can add a new class in `rotorpy/wind/` following the template there. 

As for adding more core functionality (e.g., sensors, new vehicle dynamics, animations, etc.), those require a bit more effort to make sure that all the inputs and outputs are set up accordingly. One piece of advice is that the main loop occurs in `rotorpy/simulate.py`. Under the `while` loop, you can see the process by which the vehicle dynamics, trajectory generator, IMU sensor, and controller interface with each other. 

If you are adding new functionality, as opposed to simply adding new controllers, wind fields, or trajectories, please make a new branch before starting to make those changes. 

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

This paper addresses the theory, simulation framework, and benchmarking studies.

# Elsewhere In Literature

The following is a selection of papers that have used RotorPy (or previous versions of RotorPy) for generating their simulated results: 
1. K. Y. Chee and M. A. Hsieh. "Learning-enhanced Nonlinear Model Predictive Control using Knowledge-based Neural Ordinary Differential Equations and Deep Ensembles," submitted to *5th Annual Learning for Dynamics \& Control Conference*, Philadelphia, PA USA, Jun 2023.
2. K. Y. Chee, T. Z. Jiahao and M. A. Hsieh. "KNODE-MPC: A Knowledge-Based Data-Driven Predictive Control Framework for Aerial Robots,'' in *IEEE Robotics and Automation Letters*, vol. 7, no. 2, pp. 2819-2826, Apr 2022.
3. Jiahao, Tom Z. and Chee, Kong Yao and Hsieh, M. Ani. "Online Dynamics Learning for Predictive Control with an Application to Aerial Robots," in *the Proc. of the 2022 Conference on Robot Learning (CoRL)*, Auckland, NZ, Dec 2022.
4. K. Mao, J. Welde, M. A. Hsieh, and V. Kumar, “Trajectory planning for the bidirectional quadrotor as a differentially flat hybrid system,” in *2023 International Conference on Robotics and Automation (ICRA) (accepted)*, 2023.
5. He, S., Hsu, C. D., Ong, D., Shao, Y. S., & Chaudhari, P. (2023). "Active Perception using Neural Radiance Fields," submitted to *arXiv preprint arXiv:2310.09892*.
6. Tao, R., Cheng, S., Wang, X., Wang, S., & Hovakimyan, N. (2024). "DiffTune-MPC: Closed-loop learning for model predictive control," in *IEEE Robotics and Automation Letters*.
7. Hsu, C. D., & Chaudhari, P. (2024). "Active Scout: Multi-Target Tracking Using Neural Radiance Fields in Dense Urban Environments." submitted to *arXiv preprint arXiv:2406.07431*.
8. Sanghvi, H., Folk, S., & Taylor, C. J. "OCCAM: Online Continuous Controller Adaptation with Meta-Learned Models," in *8th Annual Conference on Robot Learning (CoRL)*.


RotorPy was also listed among other UAV simulators in two recent surveys:

Dimmig, C. A., Silano, G., McGuire, K., Gabellieri, C., Hšnig, W., Moore, J., & Kobilarov, M. (2024). "Survey of Simulators for Aerial Robots: An Overview and In-Depth Systematic Comparisons," in *IEEE Robotics & Automation Magazine*.

Nikolaiev, M., & Novotarskyi, M. (2024). "Comparative Review of Drone Simulators," in *Information, Computing and Intelligent systems*, (4), 79-98.


**If you use this simulator for published work, let me know as I am happy to add your reference to this list.**

# Acknowledgements

We would like to acknowledge [Jimmy Paulos](https://github.com/jpaulos) who wrote the majority of the underlying engine for RotorPy, and the teaching assistants who contributed code to the initial version of this simulator, especially [Dan Mox](https://github.com/danmox), [Laura Jarin-Lipschitz](https://github.com/ljarin), [Rebecca Li](https://github.com/rebeccali), [Shane Rozen-Levy](https://github.com/ShaneRozenLevy), [Xu Liu](https://github.com/XuRobotics), [Yuezhan Tao](https://github.com/tyuezhan), [Yu-Ming Chen](https://github.com/yminchen), and [Fernando Cladera](https://github.com/fcladera). 

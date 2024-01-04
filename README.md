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
5. **Translational Lift** - In forward motion, the induced velocity at the rotor plane decreases, causing an increase in lift generation. *Note: currently this effect is NOT modeled in the thrust produced by the rotor.*
6. **Translational Drag** - A consequence of translational lift, and similar to **Induced Drag**, the increased lift produced in forward flight will produce an increase in induced drag on the rotor. 

Ultimately the effects boil down to forces acting anti-parallel to the relative airspeed and a combination of pitching moments acting parallel and perpendicular to the relative airspeed. The rotor aerodynamic effects (rotor drag, blade flapping, induced drag, and translational drag) can be lumped into a single drag force acting at each rotor hub, whereas parasitic drag can be lumped into a single force and moment vector acting at the center of mass. 

What's currently ignored: any lift produced by the frame or any torques produced by an imbalance of drag forces on the frame. We also currently neglect variations in the wind along the length of the UAV, implicitly assuming that the characteristic length scales of the wind fields are larger than UAV's maximum dimensions.

RotorPy also includes first-order motor dynamics to simulate lag, as well as support for spatio-temporal wind flow fields for the UAV to interact with. 

# Installation

First, clone this repository into a folder of your choosing.  

It is recommended that you use a virtual environment to install this simulator. I recommend using [Python venv](https://docs.python.org/3/library/venv.html) because it's lightweight. 

All the necessary dependencies can be installed using the following: 
**NOTE:** Read this command carefully. The period `.` is intentional. 
```
pip install -e . 
```
This will install all the packages as specified in `setup.py`. You may need to use `pip3` instead of `pip` to avoid conflict with Python 2 if installed. 

To confirm installation, you should see the package `rotorpy` listed among the other packages available in your environment. 

# Usage

#### Regular usage
A good place to start would be to reference the `rotorpy/examples/basic_usage.py` script. It goes through the necessary imports and how to create and execute an instance of the simulator. 
 
At minimum the simulator requires vehicle, controller, and trajectory objects. The vehicle (and potentially the controller) is parameterized by a unique parameter file, such as in `rotorpy/vehicles/hummingbird_params.py`. There is also the option to specify your own IMU, world bounds, and how long you would like to run the simulator for. 

The output of the simulator is a dictionary containing a time vector and the time histories of all the vehicle's states, inputs, and measurements.

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
  title={RotorPy: A Python-based Multirotor Simulator with Aerodynamics for Education and Research},
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
5. He, S., Hsu, C. D., Ong, D., Shao, Y. S., & Chaudhari, P. (2023). Active Perception using Neural Radiance Fields. *arXiv preprint arXiv:2310.09892*.

RotorPy was also listed among other UAV simulators in a recent survey:

Dimmig, C. A., Silano, G., McGuire, K., Gabellieri, C., Hönig, W., Moore, J., & Kobilarov, M. (2023). *Survey of Simulators for Aerial Robots*. arXiv preprint arXiv:2311.02296.

**If you use this simulator for published work, let me know as I am happy to add your reference to this list.**

# Acknowledgements

We would like to acknowledge [Jimmy Paulos](https://github.com/jpaulos) who wrote the majority of the underlying engine for RotorPy, and the teaching assistants who contributed code to the initial version of this simulator, especially [Dan Mox](https://github.com/danmox), [Laura Jarin-Lipschitz](https://github.com/ljarin), [Rebecca Li](https://github.com/rebeccali), [Shane Rozen-Levy](https://github.com/ShaneRozenLevy), [Xu Liu](https://github.com/XuRobotics), [Yuezhan Tao](https://github.com/tyuezhan), [Yu-Ming Chen](https://github.com/yminchen), and [Fernando Cladera](https://github.com/fcladera). 

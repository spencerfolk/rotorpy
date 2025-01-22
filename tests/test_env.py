''' 
Test the basic RotorPy environment. 
'''

from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind
from rotorpy.wind.dryden_winds import DrydenGust, DrydenGustLP
from rotorpy.wind.spatial_winds import WindTunnel
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture
from rotorpy.world import World
from rotorpy.utils.postprocessing import unpack_sim_data

import numpy as np                  
import matplotlib.pyplot as plt     
from scipy.spatial.transform import Rotation 
import os

def test_basic_env():
    

    print("\nTesting basic environment setup...")

    # Create the world
    world = World.from_file(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','rotorpy','worlds','double_pillar.json')))

    # An instance of the simulator can be generated as follows: 
    print("\tCreating the environment...")
    sim_instance = Environment(vehicle=Multirotor(quad_params),           # vehicle object, must be specified. 
                            controller=SE3Control(quad_params),        # controller object, must be specified.
                            trajectory=HoverTraj(x0=[0, 0, 0]),         # trajectory object, must be specified.
                            wind_profile=SinusoidWind(),               # OPTIONAL: wind profile object, if none is supplied it will choose no wind. 
                            sim_rate     = 100,                        # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                            world        = world,                      # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                            safety_margin= 0.25                        # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                        )

    # Setting an initial state. 
    print("\tSetting initial state...")
    x0 = {'x': np.array([0,0,0]),
        'v': np.zeros(3,),
        'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
        'w': np.zeros(3,),
        'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
        'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
    sim_instance.vehicle.initial_state = x0

    # Running the simulation
    print("\tRunning the simulation...")
    results = sim_instance.run(t_final      = 1,        # The maximum duration of the environment in seconds
                            use_mocap    = False,       # Boolean: determines if the controller should use the motion capture estimates. 
                            terminate    = False,       # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                            plot            = True,     # Boolean: plots the vehicle states and commands   
                            plot_mocap      = True,     # Boolean: plots the motion capture pose and twist measurements
                            plot_estimator  = True,     # Boolean: plots the estimator filter states and covariance diagonal elements
                            plot_imu        = True,     # Boolean: plots the IMU measurements
                            animate_bool    = True,     # Boolean: determines if the animation of vehicle state will play. 
                            animate_wind    = True,     # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV. 
                            verbose         = True,     # Boolean: will print statistics regarding the simulation. 
                            fname   = None              # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/. 
                        )

    # Save the simulation data to a CSV file
    print("\tSaving simulation data to CSV...")
    sim_instance.save_to_csv("test_simulation.csv")

    # Instead of producing a CSV, you can manually unpack the dictionary into a Pandas DataFrame using the following:
    print("\tUnpacking simulation data...") 
    dataframe = unpack_sim_data(results)
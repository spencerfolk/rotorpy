''' 
Test the RotorPy quadrotor gymnasium environment. 
'''

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward

def run_gym_environment(control_abstraction, num_drones):

    init_rotor_speed = 1788.53
    x0 = {'x': torch.zeros(num_drones,3).double(),
          'v': torch.zeros(num_drones, 3).double(),
          'q': torch.tensor([0, 0, 0, 1]).repeat(num_drones, 1).double(),
          'w': torch.zeros(num_drones, 3).double(),
          'wind': torch.zeros(num_drones, 3).double(),
          'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed]).repeat(num_drones, 1).double()}

    # First, we need to make the gym environment. The inputs to the model are as follows...
    env = QuadrotorEnv(num_drones, initial_states=x0, quad_params=quad_params, control_mode=control_abstraction)

    observation = env.reset(options={'initial_state': 'random'})

    # Unpack the observation from the environment
    state = {'x': observation[0:3], 'v': observation[3:6], 'q': observation[6:10], 'w': observation[10:13]}

    # The environment expects the control inputs to all be within the range [-1,1].
    if control_abstraction in ['cmd_vel', 'cmd_acc']:
        n_u = 3
    else:
        n_u = 4
    action = np.random.uniform(-1, 1, (num_drones, n_u))

    # Step forward in the environment
    observation, reward, terminated, truncated = env.step(action)

    # Assert that the output of the environment is as expected. 
    assert type(observation) == np.ndarray
    assert type(reward) == float or type(reward) == np.float64 or type(reward) == np.ndarray

    env.close()

def test_gym_environment():

    print("\nTesting gym environment...")

    # Test the gym environment with the different control abstractions
    print("\tTesting gym environment with control abstractions...")
    control_modes = ['cmd_motor_speeds', 'cmd_motor_thrusts', 'cmd_ctbr', 'cmd_ctbm', 'cmd_vel']
    num_drones = [1, 10]
    for control_mode in control_modes:
        for nd in num_drones:
            print(f"\t\t...control_mode: {control_mode}, num_drones: {nd}")
            run_gym_environment(control_mode, nd)
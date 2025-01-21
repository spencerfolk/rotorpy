''' 
Test the RotorPy quadrotor gymnasium environment. 
'''

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward

def run_gym_environment(control_abstraction):

    # First, we need to make the gym environment. The inputs to the model are as follows...
    env = gym.make("Quadrotor-v0", 
                    control_mode =control_abstraction, 
                    reward_fn = hover_reward,
                    quad_params = quad_params,
                    max_time = 5,
                    world = None,
                    sim_rate = 100,
                    render_mode='3D',
                    render_fps=30)

    observation, info = env.reset(options={'initial_state': 'random'})

    # Unpack the observation from the environment
    state = {'x': observation[0:3], 'v': observation[3:6], 'q': observation[6:10], 'w': observation[10:13]}

    # The environment expects the control inputs to all be within the range [-1,1].
    if control_abstraction in ['cmd_vel', 'cmd_acc']:
        n_u = 3
    else:
        n_u = 4
    action = np.random.uniform(-1, 1, n_u)

    # Step forward in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Assert that the output of the environment is as expected. 
    assert type(observation) == np.ndarray
    assert type(reward) == float or type(reward) == np.float64
    assert type(terminated) == bool
    assert type(truncated) == bool
    assert type(info) == dict

    env.close()

def test_gym_environment():

    print("\nTesting gym environment...")

    # Test the gym environment with the different control abstractions
    print("\tTesting gym environment with control abstractions...")
    control_modes = ['cmd_motor_speeds', 'cmd_motor_thrusts', 'cmd_ctbr', 'cmd_ctbm', 'cmd_vel']
    for control_mode in control_modes:
        print(f"\t\t...{control_mode}")
        run_gym_environment(control_mode)

test_gym_environment()
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward

"""
In this script, we evaluate the policy trained in ppo_hover_train.py.  

The task is for the quadrotor to stabilize to hover at the origin when starting at a random position nearby. 

"""

# First we'll set up some directories for saving the policy and logs.
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")

# Next import Stable Baselines.
try:
    import stable_baselines3
except:
    raise ImportError('To run this example you must have Stable Baselines installed via pip install stable_baselines3')

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP

num_cpu = 4   # for parallelization

# Choose the weights for our reward function. Here we are creating a lambda function over hover_reward.
reward_function = lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5})

# Make the environment. For this demo we'll train a policy in cmd_vel. Higher abstractions lead to easier tasks.
env = gym.make("Quadrotor-v0", 
                control_mode ='cmd_ctbr', 
                reward_fn = reward_function,
                quad_params = quad_params,
                max_time = 5,
                world = None,
                sim_rate = 100,
                render_mode='3D')

# from stable_baselines3.common.env_checker import check_env
# check_env(env, warn=True)  # you can check the environment using built-in tools

# Reset the environment
observation, info = env.reset(initial_state='random', options={'pos_bound': 2, 'vel_bound': 0})

# Print out policies for the user to select.
print("Select one of the models:")
models_available = os.listdir(models_dir)
for i, name in enumerate(models_available):
    print(f"{i}: {name}")
model_idx = int(input("Enter the model index: "))    

# Load the model
model_path = os.path.join(models_dir, models_available[model_idx])
print(f"Loading model from the path {model_path}")
model = PPO.load(model_path, env=env, tensorboard_log=log_dir)

num_episodes = 10
for i in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    while not terminated:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

plt.show()
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
In this script, we demonstrate how to train a hovering control policy in RotorPy using Proximal Policy Optimization. 
We use our custom quadrotor environment for Gymnasium along with stable baselines for the PPO implementation. 

The task is for the quadrotor to stabilize to hover at the origin when starting at a random position nearby. 

"""

# First we'll set up some directories for saving the policy and logs.

models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Next import Stable Baselines.
try:
    import stable_baselines3
except:
    raise ImportError('You must have Stable Baselines installed via pip install stable_baselines3')

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP
from stable_baselines3.common.evaluation import evaluate_policy     # For evaluation

# Make the environment. For this demo we'll train a policy in cmd_vel. Higher abstractions lead to easier tasks.
env = gym.make("Quadrotor-v0", 
                control_mode ='cmd_ctbr', 
                reward_fn = hover_reward,
                quad_params = quad_params,
                max_time = 5,
                world = None,
                sim_rate = 100,
                render_mode='3D')

# from stable_baselines3.common.env_checker import check_env
# check_env(env, warn=True)  # you can check the environment using built in tools

# Reset the environment
observation, info = env.reset(initial_state='random')

# Load the policy. 
model_path = os.path.join(models_dir, "PPOMlpPolicy.zip")
# If model is already trained, load it and continue training
if os.path.exists(model_path):
    model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
    print("Loading existing model.")
else:
    model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)

# Training... 
num_timesteps = 10_000
num_episodes = 100

best_mean_reward = -100_000_000  # Really large number so that in the first iteration it is overwritten

for i in range(num_episodes):

    # This line will run num_timesteps for training and log the results every so often.
    model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, tb_log_name="PPO_{}".format(i))

    if i % 10 == 0:  # Evaluate the policy every 10 episodes
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    # Save the model if it is better than the previous one
    if mean_reward >= best_mean_reward:
        best_mean_reward = mean_reward
        # Example for saving best model
        print("Saving new best model")
        model.save(model_path)
    else:
        print("Not saving model, mean reward was {:.2f}, but best reward is {:.2f}".format(mean_reward, best_mean_reward))

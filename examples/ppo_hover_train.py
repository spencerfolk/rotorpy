import os
from datetime import datetime
import torch

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv, make_default_vec_env
from rotorpy.learning.quadrotor_reward_functions import vec_hover_reward
from rotorpy.learning.learning_utils import *


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
    raise ImportError('To run this example you must have Stable Baselines installed. Try installing rotorpy with pip install rotorpy.[learning]')

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

# This script shows a simple example of training a quadrotor to hover at the origin.

# The control abstraction we will use is cmd_ctatt, for which the actions are thrusts and commanded attitude
control_mode = "cmd_ctatt"

# We will use 512 parallel environments 
num_envs = 512

device = torch.device("cpu")

# Generate random vehicle params for the quadrotors.
random_quad_params = generate_random_vehicle_params(num_envs, device, quad_params, crazyflie_randomizations)

env = make_default_vec_env(num_envs,
                           random_quad_params,
                           control_mode,
                           device,
                           render_mode="None",
                           reward_fn=vec_hover_reward)
env.reset_options["params"] = "random"
env.reset_options["vel_bound"] = 0.5
wrapped_env = VecMonitor(env)

start_time = datetime.now()

eval_env = VecMonitor(make_default_vec_env(10, quad_params, control_mode, device, render_mode="3D", reward_fn=vec_hover_reward))
checkpoint_callback = CheckpointCallback(save_freq=max(50000//num_envs, 1), save_path=f"{models_dir}/PPO/hover_{control_mode}_{start_time.strftime('%b-%d_%H-%M')}/",
                                         name_prefix='hover')
eval_callback = EvalCallback(eval_env, eval_freq=1e6//num_envs, deterministic=True, render=False)
model = PPO(MlpPolicy,
            wrapped_env,
            n_steps=32,
            batch_size=1024,
            verbose=1,
            device=device,
            tensorboard_log=log_dir)

num_timesteps = 10e6
model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False,
            tb_log_name="PPO-QuadVec_"+control_mode+"_"+ start_time.strftime('%b-%d-%H-%M'),
            callback=CallbackList([checkpoint_callback, eval_callback]))
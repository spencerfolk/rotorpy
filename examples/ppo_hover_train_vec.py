import os
from datetime import datetime
import torch

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_vec_environments import QuadrotorVecEnv, make_default_vec_env
from rotorpy.learning.quadrotor_reward_functions import vec_hover_reward_positive


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
    raise ImportError('To run this example you must have Stable Baselines installed via pip install stable_baselines3')

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

num_envs = 128
device = torch.device("cpu")
env = make_default_vec_env(num_envs, quad_params, "cmd_motor_speeds", device, render_mode="None", reward_fn=vec_hover_reward_positive)
wrapped_env = VecMonitor(env)

# eval_env = VecMonitor(make_default_vec_env(5, quad_params, "cmd_motor_speeds", device, render_mode="3D"))

model = PPO(MlpPolicy,
            wrapped_env,
            n_steps=64,
            verbose=1,
            device="cpu",
            tensorboard_log=log_dir)

num_timesteps = int(5e5)
start_time = datetime.now()
model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False,
            tb_log_name="PPO-Quad_cmd-motor_" + start_time.strftime('%H-%M-%S'))
model.save(f"{models_dir}/PPO/{start_time.strftime('%H-%M-%S')}/hover_{num_timesteps}")

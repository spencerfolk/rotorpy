import os
from datetime import datetime
import torch
import time

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_vec_environments import QuadrotorVecEnv, make_default_vec_env
from rotorpy.learning.quadrotor_reward_functions import vec_hover_reward_positive
from rotorpy.learning.learning_utils import *
from rotorpy.trajectories.hover_traj import HoverTraj

num_envs = 16
device = torch.device("cpu")
dynamics_params = BatchedDynamicsParams([quad_params for _ in range(num_envs)], num_envs, device)
env = make_default_vec_env(num_envs,
                           dynamics_params,
                           "cmd_motor_speeds",
                           device,
                           render_mode="human",
                           reward_fn=vec_hover_reward_positive)
ctrl_cmd = np.zeros((num_envs, 4))
state = env.reset()
for i in range(100):
    env.step(ctrl_cmd)
    # time.sleep(0.1)
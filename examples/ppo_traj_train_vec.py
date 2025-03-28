import os
from datetime import datetime
import torch

import rotorpy
from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_vec_environments import QuadrotorTrackingVecEnv, make_default_vec_env
from rotorpy.learning.quadrotor_reward_functions import vec_trajectory_reward
from rotorpy.learning.learning_utils import *
from rotorpy.trajectories.circular_traj import BatchedThreeDCircularTraj


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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

device = torch.device("cpu")
# Mutates the batched circle traj object in place with a new random trajectory at a specified index
def traj_randomization_fn(idx, batched_circle_traj):
    batched_circle_traj.radii[idx] = 2 * torch.rand((1,3), device=device) + 1
    batched_circle_traj.freqs[idx] = 0.5 * torch.rand((1,3), device=device) + 0.2
    batched_circle_traj.omegas[idx] = (2*np.pi*batched_circle_traj.freqs[idx]).to(device)

num_envs = 1024
init_rotor_speed = 1788.53

radii = np.ones((num_envs,3))
radii[:,2] = 0
trajectory = BatchedThreeDCircularTraj(np.zeros((num_envs,3)),
                                       radii,
                                       np.ones((num_envs, 3))*0.2,
                                       np.zeros(num_envs, dtype=bool),
                                       device=device)
x0 = {'x': torch.zeros(num_envs,3, device=device).double(),
        'v': torch.zeros(num_envs, 3, device=device).double(),
        'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_envs, 1).double(),
        'w': torch.zeros(num_envs, 3, device=device).double(),
        'wind': torch.zeros(num_envs, 3, device=device).double(),
        'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_envs, 1).double()}

reset_options = dict(rotorpy.learning.quadrotor_vec_environments.DEFAULT_RESET_OPTIONS)
reset_options["params"] = "fixed"
reset_options["pos_bound"] = 0.25
reset_options["trajectory"] = "random"

control_mode = "cmd_ctatt"
env = QuadrotorTrackingVecEnv(num_envs, 
                              initial_state=x0, 
                              trajectory=trajectory,
                              quad_params=dict(quad_params), 
                              max_time=7, 
                              control_mode=control_mode, 
                              device=device,
                              render_mode="None",
                              reward_fn=vec_trajectory_reward,
                              reset_options=reset_options,
                              traj_randomization_fn=traj_randomization_fn)

# Allows Stable Baselines to report accurate reward and episode lengths
wrapped_env = VecMonitor(env)

# Create eval environment - set up initial states and trajectory for eval. These could be different from the training env.
num_eval_envs = 5
radii = np.ones((num_eval_envs,3))
trajectory = BatchedThreeDCircularTraj(np.zeros((num_eval_envs,3)),
                                       radii,
                                       np.ones((num_eval_envs, 3))*0.2,
                                       np.zeros(num_eval_envs, dtype=bool),
                                       device=device)
x0_eval = {'x': torch.zeros(num_eval_envs,3, device=device).double(),
        'v': torch.zeros(num_eval_envs, 3, device=device).double(),
        'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_eval_envs, 1).double(),
        'w': torch.zeros(num_eval_envs, 3, device=device).double(),
        'wind': torch.zeros(num_eval_envs, 3, device=device).double(),
        'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_eval_envs, 1).double()}


eval_reset_options = dict(reset_options)
eval_reset_options["trajectory"] = "fixed"
eval_reset_options["params"] = "fixed"
eval_env = QuadrotorTrackingVecEnv(num_eval_envs, 
                              initial_state=x0_eval, 
                              trajectory=trajectory,
                              quad_params=dict(quad_params), 
                              max_time=7, 
                              control_mode=control_mode, 
                              device=device,
                              render_mode="3D",
                              reward_fn=vec_trajectory_reward,
                              reset_options=reset_options,
                              traj_randomization_fn=traj_randomization_fn)
wrapped_eval_env = VecMonitor(eval_env)


start_time = datetime.now()
checkpoint_callback = CheckpointCallback(save_freq=max(50000//num_envs, 1), save_path=f"{models_dir}/PPO/{start_time.strftime('%H-%M-%S')}/",
                                         name_prefix='hover')

eval_callback = EvalCallback(wrapped_eval_env, eval_freq=5e5//num_envs, deterministic=True, render=True)
model = PPO(MlpPolicy,
            wrapped_env,
            n_steps=64,
            batch_size=2048,
            verbose=1,
            device=device,
            tensorboard_log=log_dir)

num_timesteps = 15e6
model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False,
            tb_log_name="PPO-QuadTrajVec_"+control_mode + " " + start_time.strftime('%H-%M-%S'),
            callback=CallbackList([checkpoint_callback, eval_callback]))
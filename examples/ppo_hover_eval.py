import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy.spatial.transform import Rotation as R

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv, make_default_vec_env

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import vec_hover_reward

# For the baseline, we'll use the stock SE3 controller.
from rotorpy.controllers.quadrotor_control import SE3Control

baseline_controller = SE3Control(quad_params)

"""
In this script, we evaluate the policy trained in ppo_hover_train.py. It's meant to complement the output of ppo_hover_train.py.

The task is for the quadrotor to stabilize to hover at the origin when starting at a random position nearby. 

This script will ask the user which model they'd like to use, and then ask which specific epoch(s) they would like to evaluate.
Then, for each model epoch selected, 10 agents will be spawned alongside the baseline SE3 controller at random positions. 

Visualization is slow for this!! To speed things up, we save the figures as individual frames in data_out/ppo_hover/. If you
close out of the matplotlib figure things should run faster. You can also speed it up by only visualizing 1 or 2 RL agents. 

"""

# First we'll set up some directories for saving the policy and logs.
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies", "PPO")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")
output_dir = os.path.join(os.path.dirname(__file__), "..", "rotorpy", "data_out", "ppo_hover")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List the models here and let the user select which one. 
print("Select one of the models:")
models_available = os.listdir(models_dir)
for i, name in enumerate(models_available):
    print(f"{i}: {name}")
model_idx = int(input("Enter the model index: "))  
num_timesteps_dir = os.path.join(models_dir, models_available[model_idx])

# Next import Stable Baselines.
try:
    import stable_baselines3
except:
    raise ImportError('To run this example you must have Stable Baselines installed via pip install stable_baselines3')

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP

# Choose the weights for our reward function. Here we are creating a lambda function over hover_reward.
# reward_function = lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5})
# reward_function = lambda obs, act: vec_hover_reward_positive(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5})

control_mode = "cmd_ctatt"

# Set up the figure for plotting all the agents.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make the environments for the RL agents.
# +1 for the baseline SE3 controller.
num_quads = 10 + 1
device = torch.device("cpu")

env = make_default_vec_env(num_quads, quad_params, control_mode, device, render_mode="3D", reward_fn=vec_hover_reward, fig=fig, ax=ax)
env.reset_options["initial_states"] = "random"
env.reset_options["params"] = "fixed"

# Print out policies for the user to select.
def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])
num_timesteps_list = [fname for fname in os.listdir(num_timesteps_dir) if fname.startswith('hover_')]
num_timesteps_list_sorted = sorted(num_timesteps_list, key=extract_number)

print("Select one of the epochs:")
for i, name in enumerate(num_timesteps_list_sorted):
    print(f"{i}: {name}")
num_timesteps_idxs = [int(input("Enter the epoch index: "))]  

# You can optionally just hard code a series of epochs you'd like to evaluate all at once.
# e.g. num_timesteps_idxs = [0, 1, 2, ...]

# Evaluation...
for (k, num_timesteps_idx) in enumerate(num_timesteps_idxs):  # For each num_timesteps index...

    print(f"[ppo_hover_eval.py]: Starting epoch {k+1} out of {len(num_timesteps_idxs)}.")

    # Load the model for the appropriate epoch.
    model_path = os.path.join(num_timesteps_dir, num_timesteps_list_sorted[num_timesteps_idx])
    print(f"Loading model from the path {model_path}")
    model = PPO.load(model_path, env=env, tensorboard_log=log_dir)

    # Set figure title for 3D plot.
    fig.suptitle(f"Model: PPO/{models_available[model_idx]}, Num Timesteps: {extract_number(num_timesteps_list_sorted[num_timesteps_idx]):,}")

    # Visualization is slow, so we'll also save frames to make a GIF later.
    # Set the path for these frames here. 
    frame_path = os.path.join(output_dir, num_timesteps_list_sorted[num_timesteps_idx][:-4])
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    # Collect observations for each environment.
    observation = env.reset()

    # This is a list of env termination conditions so that the loop only ends when the final env is terminated. 
    terminated = np.array([False for _ in range(num_quads)])

    # Arrays for plotting position vs time. 
    T = [0]
    pos = observation[:,0:3].reshape(1, num_quads, 3)

    j = 0  # Index for frames. Only updated when the last environment runs its update for the time step. 
    while not np.all(terminated):
        frames = []  # Reset frames. 
        env.render()

        # Get the policy's actions
        action, _ = model.predict(observation, deterministic=True)

        # Run the SE3 controller for the last drone in the environment.
        state = {'x': observation[-1, 0:3], 
                 'v': observation[-1, 3:6], 
                 'q': observation[-1, 6:10], 
                 'w': observation[-1, 10:13]}
        flat = {'x': [0, 0, 0], 
                'x_dot': [0, 0, 0], 
                'x_ddot': [0, 0, 0], 
                'x_dddot': [0, 0, 0],
                'yaw': 0, 
                'yaw_dot': 0, 
                'yaw_ddot': 0}

        control_dict = baseline_controller.update(0, state, flat)

        # Rescale the controller actions to [-1, 1], and convert quaternions to euler angles, expected by environment
        ctrl_norm_thrust = (control_dict["cmd_thrust"] - 4 * env.min_thrust[-1]) / (4 * env.max_thrust[-1] - 4 * env.min_thrust[-1])
        ctrl_norm_thrust = ctrl_norm_thrust * 2 - 1
        eulers = R.from_quat(control_dict["cmd_q"]).as_euler('xyz')
        eulers_norm = 2 * (eulers + np.pi) / (2 * np.pi) - 1
        ctrlr_action = np.hstack([ctrl_norm_thrust, eulers_norm])

        # Replace the last action with the SE3 controller action.
        action[-1] = ctrlr_action

        observation, reward, done, info = env.step(action)
        terminated = np.logical_or(done, terminated)

        pos = np.append(pos, observation[:,0:3].reshape(1, num_quads, 3), axis=0)
        pos[-1, terminated, :] = 0

        # Keep in mind that the environment will automatically reset any quadrotors that are terminated or truncated, as per the Stable Baselines expectation.
        T.append(env.t[0])

        if env.render_mode == "3D":
            frame = os.path.join(frame_path, 'frame_'+str(j)+'.png')
            fig.savefig(frame)
            j += 1

    T = np.array(T)[:-1]

    # Plot position vs time. 
    fig_pos, ax_pos = plt.subplots(nrows=3, ncols=1, num="Position vs Time")
    fig_pos.suptitle(f"Model: PPO/{models_available[model_idx]}, Num Timesteps: {extract_number(num_timesteps_list_sorted[num_timesteps_idx]):,}")
    ax_pos[0].plot(T, pos[:-1, 0, 0], 'b-', linewidth=1, label="RL")
    ax_pos[0].plot(T, pos[:-1, 1:-1, 0], 'b-', linewidth=1)
    ax_pos[0].plot(T, pos[:-1, -1, 0], 'k-', linewidth=2, label="GC")
    ax_pos[0].legend()
    ax_pos[0].set_ylabel("X, m")
    ax_pos[0].set_ylim([-2.5, 2.5])
    ax_pos[1].plot(T, pos[:-1, 0, 1], 'b-', linewidth=1, label="RL")
    ax_pos[1].plot(T, pos[:-1, 1:-1, 1], 'b-', linewidth=1)
    ax_pos[1].plot(T, pos[:-1, -1, 1], 'k-', linewidth=2, label="GC")
    ax_pos[1].set_ylabel("Y, m")
    ax_pos[1].set_ylim([-2.5, 2.5])
    ax_pos[2].plot(T, pos[:-1, 0, 2], 'b-', linewidth=1, label="RL")
    ax_pos[2].plot(T, pos[:-1, 1:-1, 2], 'b-', linewidth=1)
    ax_pos[2].plot(T, pos[:-1, -1, 2], 'k-', linewidth=2, label="GC")
    ax_pos[2].set_ylabel("Z, m")
    ax_pos[2].set_ylim([-2.5, 2.5])
    ax_pos[2].set_xlabel("Time, s")

    # Save fig. 
    fig_pos.savefig(os.path.join(frame_path, 'position_vs_time.png'))

plt.show()

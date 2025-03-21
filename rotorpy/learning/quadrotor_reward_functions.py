import numpy as np
from scipy.spatial.transform import Rotation
import torch

import gymnasium as gym
from gymnasium import spaces

import math

"""
Reward functions for quadrotor tasks. 
"""

def hover_reward(observation, action, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and 
    action reward.
    """

    # Compute the distance to goal
    dist_reward = -weights['x']*np.linalg.norm(observation[0:3])

    # Compute the velocity reward
    vel_reward = -weights['v']*np.linalg.norm(observation[3:6])

    # Compute the angular rate reward
    ang_rate_reward = -weights['w']*np.linalg.norm(observation[10:13])

    # Compute the action reward
    action_reward = -weights['u']*np.linalg.norm(action)

    return dist_reward + vel_reward + action_reward + ang_rate_reward


def hover_reward_positive(observation, action, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and
    action reward.
    """

    # Compute the distance to goal
    dist_reward = weights['x']*1/(1+np.linalg.norm(observation[0:3]))

    # Compute the velocity reward
    vel_reward = weights['v']*1/(1+np.linalg.norm(observation[3:6]))

    # Compute the angular rate reward
    ang_rate_reward = weights['w']*1/(1+np.linalg.norm(observation[10:13]))

    # Compute the action reward
    action_reward = weights['u']*1/(1+np.linalg.norm(action))

    return dist_reward + vel_reward + action_reward + ang_rate_reward


def vec_hover_reward(observation, action, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and
    action reward. Computes rewards for each environment.
    """

    # Compute the distance to goal
    dist_reward = -weights['x']*np.linalg.norm(observation[...,0:3], axis=-1)

    # Compute the velocity reward
    vel_reward = -weights['v']*np.linalg.norm(observation[...,3:6], axis=-1)

    # Compute the angular rate reward
    ang_rate_reward = -weights['w']*np.linalg.norm(observation[...,10:13], axis=-1)

    # Compute the action reward
    action_reward = -weights['u']*np.linalg.norm(action, axis=-1)

    return dist_reward + vel_reward + action_reward + ang_rate_reward


def vec_hover_reward_positive(observation, action, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and
    action reward. Computes rewards for each environment.
    """

    # distance reward - reward getting closer to 0
    dist_reward = weights['x'] * 1/(1+np.linalg.norm(observation[...,0:3], axis=-1))

    # Compute the velocity reward
    vel_reward = weights['v'] * 1/(1+np.linalg.norm(observation[...,3:6], axis=-1))

    # Compute the angular rate reward
    ang_rate_reward = weights['w']*1/(1+np.linalg.norm(observation[...,10:13], axis=-1))

    # Compute the action reward
    action_reward = weights['u']*1/(1+np.linalg.norm(action, axis=-1))

    return dist_reward + vel_reward + action_reward + ang_rate_reward


def vec_trajectory_reward(observation, action, weights={'x':1, 'v':0.5, 'yaw':0.1, 'yaw_dot':0.1, 'w':0.01}):
    """
    Rewards following a trajectory. It is a combination of position error, velocity error, yaw error, yaw rate error, 
    and body rates.
    """

    # Compute the distance to goal
    dist_reward = weights['x']*1/(1+np.linalg.norm(observation[...,0:3] - observation[...,13:16], axis=-1))

    # Compute the velocity reward
    vel_reward = weights['v']*1/(1+np.linalg.norm(observation[...,3:6] - observation[...,16:19], axis=-1))

    # Compute the yaw error reward
    yaw_reward = weights['yaw']*1/(1+np.abs(observation[...,6] - observation[...,19]))

    # Compute the yaw rate error reward
    yaw_dot_reward = weights['yaw_dot']*1/(1+np.abs(observation[...,7] - observation[...,20]))

    # Compute the angular rate reward
    ang_rate_reward = weights['w']*1/(1+np.linalg.norm(observation[...,10:13], axis=-1))

    return dist_reward + vel_reward + yaw_reward + yaw_dot_reward + ang_rate_reward
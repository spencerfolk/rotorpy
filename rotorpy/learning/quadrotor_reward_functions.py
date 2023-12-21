import numpy as np
from scipy.spatial.transform import Rotation

import gymnasium as gym
from gymnasium import spaces

import math

"""
Reward functions for quadrotor tasks. 
"""

def hover_reward(observation, action):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and 
    action reward.
    """

    dist_weight = 1
    vel_weight = 0.1
    action_weight = 0.001
    ang_rate_weight = 0.1

    # Compute the distance to goal
    dist_reward = -dist_weight*np.linalg.norm(observation[0:3])

    # Compute the velocity reward
    vel_reward = -vel_weight*np.linalg.norm(observation[3:6])

    # Compute the angular rate reward
    ang_rate_reward = -ang_rate_weight*np.linalg.norm(observation[10:13])

    # Compute the action reward
    action_reward = -action_weight*np.linalg.norm(action)

    return dist_reward + vel_reward + action_reward + ang_rate_reward
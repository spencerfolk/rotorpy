import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# For this demonstration, we'll just use the SE3 controller. 
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params

controller = SE3Control(quad_params)

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward

# First, we need to make the gym environment. The inputs to the model are as follows...
"""
Inputs:
    initial_state: the initial state of the quadrotor. The default is hover. 
    control_mode: the appropriate control abstraction that is used by the controller, options are...
            'cmd_motor_speeds': the controller directly commands motor speeds. 
            'cmd_motor_thrusts': the controller commands forces for each rotor.
            'cmd_ctbr': the controller commands a collective thrsut and body rates. 
            'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
            'cmd_vel': the controller commands a velocity vector in the body frame. 
    reward_fn: the reward function, default to hover, but the user can pass in any function that is used as a reward. 
    quad_params: the parameters for the quadrotor. 
    max_time: the maximum time of the session. 
    world: the world for the quadrotor to operate within. 
    sim_rate: the simulation rate (in Hz), i.e. the timestep. 
    render_mode: render the quadrotor.
            'None': no rendering
            'console': output text describing the environment. 
            '3D': will render the quadrotor in 3D. WARNING: THIS IS SLOW. 

"""
env = gym.make("Quadrotor-v0", 
                control_mode ='cmd_motor_speeds', 
                reward_fn = hover_reward,
                quad_params = quad_params,
                max_time = 5,
                world = None,
                sim_rate = 100,
                render_mode='3D',
                render_fps=30)

# Now reset the quadrotor.
# Setting initial_state to 'random' will randomly place the vehicle in the map near the origin.
# But you can also set the environment resetting to be deterministic. 
observation, info = env.reset(options={'initial_state': 'random'})

# Number of timesteps
T = 300
time = np.arange(T)*(1/100)      # Just for plotting purposes.
position = np.zeros((T, 3))      # Just for plotting purposes. 
velocity = np.zeros((T, 3))      # Just for plotting purposes.
reward_sum = np.zeros((T,))      # Just for plotting purposes.
actions = np.zeros((T, 4))       # Just for plotting purposes.

for i in range(T):

    ##### Below is just code for computing the action via the SE3 controller and converting it to an action [-1,1]

    # Unpack the observation from the environment
    state = {'x': observation[0:3], 'v': observation[3:6], 'q': observation[6:10], 'w': observation[10:13]}
    
    # For illustrative purposes, just command the quad to hover.
    flat = {'x': [0, 0, 0], 
            'x_dot': [0, 0, 0], 
            'x_ddot': [0, 0, 0], 
            'x_dddot': [0, 0, 0],
            'yaw': 0, 
            'yaw_dot': 0, 
            'yaw_ddot': 0}
    control_dict = controller.update(0, state, flat)

    # Extract the commanded motor speeds.
    cmd_motor_speeds = control_dict['cmd_motor_speeds']

    # The environment expects the control inputs to all be within the range [-1,1]
    action = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])

    ###### Alternatively, we could just randomly sample the action space. 
#     action = np.random.uniform(low=-1, high=1, size=(4,))

    # Step forward in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # For plotting, save the relevant information
    position[i, :] = observation[0:3]
    velocity[i, :] = observation[3:6]
    if i == 0:
        reward_sum[i] = reward
    else:
        reward_sum[i] = reward_sum[i-1] + reward
    actions[i, :] = action

env.close()

# Plotting

(fig, axes) = plt.subplots(nrows=2, ncols=1, num='Quadrotor State')
ax = axes[0]
ax.plot(time, position[:, 0], 'r', label='X')
ax.plot(time, position[:, 1], 'g', label='Y')
ax.plot(time, position[:, 2], 'b', label='Z')
ax.set_ylabel("Position, m")
ax.legend()

ax = axes[1]
ax.plot(time, velocity[:, 0], 'r', label='X')
ax.plot(time, velocity[:, 1], 'g', label='Y')
ax.plot(time, velocity[:, 2], 'b', label='Z')
ax.set_ylabel("Velocity, m/s")
ax.set_xlabel("Time, s")

(fig, axes) = plt.subplots(nrows=2, ncols=1, num="Action and Reward")
ax = axes[0]
ax.plot(time, actions[:, 0], 'r', label='action 1')
ax.plot(time, actions[:, 1], 'g', label='action 2')
ax.plot(time, actions[:, 2], 'b', label='action 3')
ax.plot(time, actions[:, 3], 'm', label='action 4')
ax.set_ylabel("Action")
ax.legend()

ax = axes[1]
ax.plot(time, reward_sum, 'k')
ax.set_xlabel("Time, s")
ax.set_ylabel("Reward Sum")

plt.show()
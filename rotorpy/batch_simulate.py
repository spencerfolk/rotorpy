from enum import Enum
import copy
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import roma
import torch
from simulate import ExitStatus


# TODO(hersh500): finish this
def simulate_batch(world,
                   initial_states,
                   vehicles,
                   controller,
                   trajectories,
                   wind_profile,
                   t_final,
                   t_step,
                   safety_margin,
                   terminate=None):
    """
    Perform a vehicle simulation and return the numerical results.
    Note that, currently, compared to the normal simulate() function, simulate_batch() does not support
    IMU measurements, mocap, or the state estimator.

    Inputs:
        world, a class representing the world it is flying in, including objects and world bounds. 
        initial_state, a dict defining the vehicle initial conditions with appropriate keys
        vehicle, Vehicle object containing the dynamics
        controller, Controller object containing the controller
        trajectory, Trajectory object containing the trajectory to follow
        wind_profile, Wind Profile object containing the wind generator. 
        t_final, maximum duration of simulation, s
        t_step, the time between each step in the simulator, s
        safety_margin, the radius of the ball surrounding the vehicle position to determine if a collision occurs
        imu, IMU object that generates accelerometer and gyroscope readings from the vehicle state
        terminate, None, False, or a function of time and state that returns
            ExitStatus. If None (default), terminate when hover is reached at
            the location of trajectory with t=inf. If False, never terminate
            before timeout or error. If a function, terminate when returns not
            None.

    Outputs:
        time, seconds, shape=(N,)
        state, a dict describing the state history with keys
            x, position, m, shape=(N,B,3)
            v, linear velocity, m/s, shape=(N,B,3)
            q, quaternion [i,j,k,w], shape=(N,B,4)
            w, angular velocity, rad/s, shape=(N,B,3)
            rotor_speeds, motor speeds, rad/s, shape=(N,B,n) where n is the number of rotors
            wind, wind velocity, m/s, shape=(N,B,3)
        control, a dict describing the command input history with keys
            cmd_motor_speeds, motor speeds, rad/s, shape=(N,B,4)
            cmd_q, commanded orientation (not used by simulator), quaternion [i,j,k,w], shape=(N,B,4)
            cmd_w, commanded angular velocity (not used by simulator), rad/s, shape=(N,B,3)
        flat, a dict describing the desired flat outputs from the trajectory with keys
            x,        position, m
            x_dot,    velocity, m/s
            x_ddot,   acceleration, m/s**2
            x_dddot,  jerk, m/s**3
            x_ddddot, snap, m/s**4
            yaw,      yaw angle, rad
            yaw_dot,  yaw rate, rad/s
        exit_times, an array indicating at which timestep (not time!) each vehicle in the batch completed its sim, shape = (B)
        exit_status, an ExitStatus enum indicating the reason for termination.
    """

    # Coerce entries of initial state into numpy arrays, if they are not already.
    # initial_states = {k: np.array(v) for k, v in initial_states.items()}
    assert(torch.is_tensor(initial_states[k]) for k in initial_states.keys())

    if terminate is None:    # Default exit. Terminate at final position of trajectory.
        normal_exit = traj_end_exit(initial_states, trajectories, using_vio = False)
    elif terminate is False: # Never exit before timeout.
        normal_exit = lambda t, s: None
    else:                    # Custom exit.
        normal_exit = terminate

    time    = [0]
    state   = [copy.deepcopy(initial_states)]
    imu_measurements = []
    mocap_measurements = []
    imu_gt = []
    state_estimate = []
    flat    = [trajectories.update(time[-1])]
    control = [controller.update(time[-1], state[-1], flat[-1])]
    state_dot =  vehicles.statedot(state[0], control[0], t_step)

    exit_status = None

    while True:
        # how slow is this?
        exit_status = exit_status or safety_exit(world, safety_margin, state[-1], flat[-1], control[-1])
        exit_status = exit_status or normal_exit(time[-1], state[-1])
        exit_status = exit_status or time_exit(time[-1], t_final)
        if exit_status:
            break
        time.append(time[-1] + t_step)
        state[-1]['wind'] = wind_profile.update(time[-1], state[-1]['x'])
        state.append(vehicles.step(state[-1], control[-1], t_step))
        flat.append(trajectories.update(time[-1]))
        control.append(controller.update(time[-1], state[-1], flat[-1]))
        state_dot = vehicles.statedot(state[-1], control[-1], t_step)

    time    = np.array(time, dtype=float)    
    state   = merge_dicts(state)
    mocap_measurements = merge_dicts(mocap_measurements)
    control         = merge_dicts(control)
    flat            = merge_dicts(flat)
    state_estimate  = merge_dicts(state_estimate)

    return (time, state, control, flat, imu_measurements, imu_gt, mocap_measurements, state_estimate, exit_status)

def merge_dicts(dicts_in):
    """
    Concatenates contents of a list of N state dicts into a single dict by
    prepending a new dimension of size N. This is more convenient for plotting
    and analysis. Requires dicts to have consistent keys and have values that
    are numpy arrays.
    """
    dict_out = {}
    for k in dicts_in[0].keys():
        dict_out[k] = []
        for d in dicts_in:
            dict_out[k].append(d[k])
        dict_out[k] = np.array(dict_out[k])
    return dict_out


# TODO(hersh500): make this work in the batched case - I think this requires some care
def traj_end_exit(initial_state, trajectory, using_vio = False):
    """
    Returns a exit function. The exit function returns an exit status message if
    the quadrotor is near hover at the end of the provided trajectory. If the
    initial state is already at the end of the trajectory, the simulation will
    run for at least one second before testing again.
    """

    xf = trajectory.update(np.inf)['x']
    yawf = trajectory.update(np.inf)['yaw']  # (num_drones, 1)
    # rotf = Rotation.from_rotvec(yawf * np.array([0, 0, 1])) # create rotation object that describes yaw
    # I suspect there will be some broadcasting issues here.
    rotf = roma.rotvec_to_rotmat(yawf*torch.tensor([0,0,1]).unsqueeze(0))
    if np.array_equal(initial_state['x'], xf):
        min_time = 1.0
    else:
        min_time = 0

    def exit_fn(time, state):
        cur_attitude = Rotation.from_quat(state['q'])
        err_attitude = rotf * cur_attitude.inv() # Rotation between current and final
        angle = norm(err_attitude.as_rotvec()) # angle in radians from vertical
        # Success is reaching near-zero speed with near-zero position error.
        if using_vio:
            # set larger threshold for VIO due to noisy measurements
            if time >= min_time and norm(state['x'] - xf) < 1 and norm(state['v']) <= 1 and angle <= 1:
                return ExitStatus.COMPLETE
        else:
            if time >= min_time and norm(state['x'] - xf) < 0.02 and norm(state['v']) <= 0.03 and angle <= 0.02:
                return ExitStatus.COMPLETE
        return None
    return exit_fn

def time_exit(time, t_final):
    """
    Return exit status if the time exceeds t_final, otherwise None.
    """
    if time >= t_final:
        return ExitStatus.TIMEOUT
    return None

def safety_exit(world, margin, state, flat, control):
    """
    Return exit status if any safety condition is violated, otherwise None.
    """
    if np.any(np.abs(state['v']) > 20):
        return ExitStatus.OVER_SPEED
    if np.any(np.abs(state['w']) > 100):
        return ExitStatus.OVER_SPIN

    if len(world.world.get('blocks', [])) > 0:
        # If a world has objects in it we need to check for collisions.  
        collision_pts = world.path_collisions(state['x'], margin)
        no_collision = collision_pts.size == 0
        if not no_collision:
            return ExitStatus.COLLISION
    return None
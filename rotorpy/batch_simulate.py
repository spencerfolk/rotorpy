from enum import Enum
import copy
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import roma
import torch

import rotorpy.wind.default_winds
from rotorpy.controllers.quadrotor_control import SE3ControlBatch
from rotorpy.vehicles.batched_multirotor import BatchedMultirotor
from rotorpy.simulate import ExitStatus

# exitstatus_array = np.array([0, ExitStatus.COMPLETE, ExitStatus.TIMEOUT, ExitStatus.INF_VALUE,
#                              ExitStatus.NAN_VALUE, ExitStatus.OVER_SPEED, ExitStatus.OVER_SPIN,
#                              ExitStatus.FLY_AWAY, ExitStatus.COLLISION])

def simulate_batch(world,
                   initial_states,
                   vehicles: BatchedMultirotor,
                   controller: SE3ControlBatch,
                   trajectories,
                   wind_profile,
                   t_final,
                   t_step,
                   safety_margin,
                   terminate=None,
                   start_times=None):
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
            0.

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

    # Coerce entries of initial state into torch arrays, if they are not already.
    # initial_states = {k: np.array(v) for k, v in initial_states.items()}
    assert(torch.is_tensor(initial_states[k]) for k in initial_states.keys())
    if wind_profile is None:
        wind_profile = rotorpy.wind.default_winds.NoWind(vehicles.num_drones)
    assert(wind_profile.num_drones == vehicles.num_drones)
    if len(world.world['blocks']) > 0:
        raise Warning("Batched simulation does not check for collisions.")
    t_final = np.array(t_final)

    if terminate is None:    # Default exit. Terminate at final position of trajectory.
        normal_exit = traj_end_exit(initial_states, trajectories, using_vio = False)
    elif terminate is False: # Never exit before timeout.
        normal_exit = lambda t, s: None
    else:                    # Custom exit.
        normal_exit = terminate

    if start_times is None:
        time = [np.zeros(vehicles.num_drones)]
    else:
        time = [start_times]
    exit_status = np.array([None] * vehicles.num_drones)
    done = np.zeros(vehicles.num_drones, dtype=bool)
    running_idxs = np.arange(vehicles.num_drones)
    done_times = np.zeros(vehicles.num_drones, dtype=int)
    state   = [copy.deepcopy(initial_states)]
    imu_measurements = []
    mocap_measurements = []
    imu_gt = []
    state_estimate = []
    flat    = [trajectories.update(time[-1])]
    control = [controller.update(time[-1], state[-1], flat[-1], idxs=None)]
    # state_dot =  vehicles.statedot(state[0], control[0], t_step)
    step = 0

    # how to resolve how to compute exit status??
    while True:
        prev_status = np.array(done, dtype=bool)
        se = safety_exit(world, safety_margin, state[-1], flat[-1], control[-1])
        ne = normal_exit(time[-1], state[-1])
        te = time_exit(time[-1], t_final)
        exit_status[running_idxs] = np.where(se[running_idxs], ExitStatus.OVER_SPEED, None)  # Not exactly correct.
        exit_status[running_idxs] = np.where(ne[running_idxs], ExitStatus.COMPLETE, None)
        exit_status[running_idxs] = np.where(te[running_idxs], ExitStatus.TIMEOUT, None)

        done = np.logical_or(done, se)
        done = np.logical_or(done, ne)
        done = np.logical_or(done, te)
        done_this_iter = np.logical_xor(prev_status, done)
        done_times[done_this_iter] = step
        if np.all(done):
            break
        running_idxs = np.nonzero(np.logical_not(done))[0]

        time.append(time[-1] + t_step)
        state[-1]['wind'] = wind_profile.update(time[-1], state[-1]['x'])
        state.append(vehicles.step(state[-1], control[-1], t_step, idxs=running_idxs.flatten()))
        flat.append(trajectories.update(time[-1]))
        control.append(controller.update(time[-1], state[-1], flat[-1], idxs=running_idxs.flatten()))
        # state_dot = vehicles.statedot(state[-1], control[-1], t_step)
        step += 1

    time    = np.array(time, dtype=float)
    state   = merge_dicts(state)
    control         = merge_dicts(control)
    flat            = merge_dicts(flat)

    return (time, state, control, flat, exit_status, done_times)


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
            dict_out[k].append(d[k].cpu().numpy())
        dict_out[k] = np.array(dict_out[k])
    return dict_out


def traj_end_exit(initial_state, trajectory, using_vio = False):
    """
    Returns a exit function. The exit function returns an exit status message if
    the quadrotor is near hover at the end of the provided trajectory. If the
    initial state is already at the end of the trajectory, the simulation will
    run for at least one second before testing again.
    """

    xf = trajectory.update(np.inf)['x']
    yawf = trajectory.update(np.inf)['yaw'].unsqueeze(-1)  # (num_drones, 1)
    rotf = roma.rotvec_to_rotmat(yawf*torch.tensor([0,0,1]).unsqueeze(0))
    min_times = torch.all(torch.eq(initial_state['x'], xf), dim=-1).float()

    def exit_fn(time, state):
        cur_attitudes = roma.unitquat_to_rotmat(state['q'])
        err_attitudes = rotf * torch.linalg.inv(cur_attitudes)
        angle = torch.linalg.norm(roma.rotmat_to_rotvec(err_attitudes))
        # Success is reaching near-zero speed with near-zero position error.
        if using_vio:
            # set larger threshold for VIO due to noisy measurements
            # can't pass multiple arrays into torch.logical_and()
            cond1 = torch.logical_and(torch.from_numpy(time) >= min_times, torch.linalg.norm(state['x'] - xf, dim=-1).cpu() < 1)
            cond2 = torch.logical_and(torch.linalg.norm(state['v'], dim=-1).cpu() <= 1, angle <= 1)
            cond = torch.logical_and(cond1, cond2).cpu().numpy()
        else:
            cond1 = torch.logical_and(torch.from_numpy(time) >= min_times, torch.linalg.norm(state['x'] - xf, dim=-1).cpu() < 0.02)
            cond2 = torch.logical_and(torch.linalg.norm(state['v'], dim=-1).cpu() <= 0.02, angle <= 0.02)
            cond = torch.logical_and(cond1, cond2).cpu().numpy()
        return np.where(cond, True, False)
    return exit_fn

def time_exit(times: np.ndarray, t_finals: np.ndarray):
    """
    Return exit status if the time exceeds t_final, otherwise None.
    """
    # return np.where(times >= t_finals)
    # if time >= t_final:
    #     return ExitStatus.TIMEOUT
    # return None
    return np.where(times >= t_finals, True, False)

def safety_exit(world, margin, state, flat, control):
    """
    Return exit status per drone if their safety conditions is violated, otherwise 0.
    """
    status = np.zeros(state['x'].shape[0], dtype=bool)
    status = np.where(np.any(np.abs(state['v'].cpu().numpy()) > 20, axis=-1),
                      True,
                      status)
    status = np.where(np.any(np.abs(state['w'].cpu().numpy()) > 100, axis=-1),
                      True,
                      status)
    return status

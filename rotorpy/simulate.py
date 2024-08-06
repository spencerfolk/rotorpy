from enum import Enum
import copy
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

class ExitStatus(Enum):
    """ Exit status values indicate the reason for simulation termination. """
    COMPLETE     = 'Success: End reached.'
    TIMEOUT      = 'Timeout: Simulation end time reached.'
    INF_VALUE    = 'Failure: Your controller returned inf motor speeds.'
    NAN_VALUE    = 'Failure: Your controller returned nan motor speeds.'
    OVER_SPEED   = 'Failure: Your quadrotor is out of control; it is going faster than 100 m/s. The Guinness World Speed Record is 73 m/s.'
    OVER_SPIN    = 'Failure: Your quadrotor is out of control; it is spinning faster than 100 rad/s. The onboard IMU can only measure up to 52 rad/s (3000 deg/s).'
    FLY_AWAY     = 'Failure: Your quadrotor is out of control; it flew away with a position error greater than 20 meters.'
    COLLISION    = 'Failure: Your quadrotor collided with an object.'

def simulate(world, initial_state, vehicle, controller, trajectory, wind_profile, imu, mocap, estimator, t_final, t_step, safety_margin, use_mocap, terminate=None):
    """
    Perform a vehicle simulation and return the numerical results.

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
        mocap, a MotionCapture object that provides noisy measurements of pose and twist with artifacts. 
        use_mocap, a boolean to determine in noisy measurements from mocap should be used for quadrotor control
        estimator, an estimator object that provides estimates of a portion or all of the vehicle state.

    Outputs:
        time, seconds, shape=(N,)
        state, a dict describing the state history with keys
            x, position, m, shape=(N,3)
            v, linear velocity, m/s, shape=(N,3)
            q, quaternion [i,j,k,w], shape=(N,4)
            w, angular velocity, rad/s, shape=(N,3)
            rotor_speeds, motor speeds, rad/s, shape=(N,n) where n is the number of rotors
            wind, wind velocity, m/s, shape=(N,3)
        control, a dict describing the command input history with keys
            cmd_motor_speeds, motor speeds, rad/s, shape=(N,4)
            cmd_q, commanded orientation (not used by simulator), quaternion [i,j,k,w], shape=(N,4)
            cmd_w, commanded angular velocity (not used by simulator), rad/s, shape=(N,3)
        flat, a dict describing the desired flat outputs from the trajectory with keys
            x,        position, m
            x_dot,    velocity, m/s
            x_ddot,   acceleration, m/s**2
            x_dddot,  jerk, m/s**3
            x_ddddot, snap, m/s**4
            yaw,      yaw angle, rad
            yaw_dot,  yaw rate, rad/s
        imu_measurements, a dict containing the biased and noisy measurements from an accelerometer and gyroscope
            accel,  accelerometer, m/s**2
            gyro,   gyroscope, rad/s
        imu_gt, a dict containing the ground truth (no noise, no bias) measurements from an accelerometer and gyroscope
            accel,  accelerometer, m/s**2
            gyro,   gyroscope, rad/s
        mocap_measurements, a dict containing noisy measurements of pose and twist for the vehicle. 
            x, position (inertial)
            v, velocity (inertial)
            q, orientation of body w.r.t. inertial frame.
            w, body rates in the body frame. 
        exit_status, an ExitStatus enum indicating the reason for termination.
    """

    # Coerce entries of initial state into numpy arrays, if they are not already.
    initial_state = {k: np.array(v) for k, v in initial_state.items()}

    if terminate is None:    # Default exit. Terminate at final position of trajectory.
        normal_exit = traj_end_exit(initial_state, trajectory, using_vio = False)
    elif terminate is False: # Never exit before timeout.
        normal_exit = lambda t, s: None
    else:                    # Custom exit.
        normal_exit = terminate

    time    = [0]
    state   = [copy.deepcopy(initial_state)]
    state[0]['wind'] = wind_profile.update(0, state[0]['x'])   # TODO: move this line elsewhere so that other objects that don't have wind as a state can work here. 
    imu_measurements = []
    mocap_measurements = []
    imu_gt = []
    state_estimate = []
    flat    = [trajectory.update(time[-1])]
    mocap_measurements.append(mocap.measurement(state[-1], with_noise=True, with_artifacts=False))
    if use_mocap:
        # In this case the controller will use the motion capture estimate of the pose and twist for control. 
        control = [controller.update(time[-1], mocap_measurements[-1], flat[-1])]
    else:
        control = [controller.update(time[-1], state[-1], flat[-1])]
    state_dot =  vehicle.statedot(state[0], control[0], t_step)
    imu_measurements.append(imu.measurement(state[-1], state_dot, with_noise=True))
    imu_gt.append(imu.measurement(state[-1], state_dot, with_noise=False))
    state_estimate.append(estimator.step(state[0], control[0], imu_measurements[0], mocap_measurements[0]))

    exit_status = None

    while True:
        exit_status = exit_status or safety_exit(world, safety_margin, state[-1], flat[-1], control[-1])
        exit_status = exit_status or normal_exit(time[-1], state[-1])
        exit_status = exit_status or time_exit(time[-1], t_final)
        if exit_status:
            break
        time.append(time[-1] + t_step)
        state[-1]['wind'] = wind_profile.update(time[-1], state[-1]['x'])
        state.append(vehicle.step(state[-1], control[-1], t_step))
        flat.append(trajectory.update(time[-1]))
        mocap_measurements.append(mocap.measurement(state[-1], with_noise=True, with_artifacts=mocap.with_artifacts))
        state_estimate.append(estimator.step(state[-1], control[-1], imu_measurements[-1], mocap_measurements[-1]))
        if use_mocap:
            control.append(controller.update(time[-1], mocap_measurements[-1], flat[-1]))
        else:
            control.append(controller.update(time[-1], state[-1], flat[-1]))
        state_dot = vehicle.statedot(state[-1], control[-1], t_step)
        imu_measurements.append(imu.measurement(state[-1], state_dot, with_noise=True))
        imu_gt.append(imu.measurement(state[-1], state_dot, with_noise=False))

    time    = np.array(time, dtype=float)    
    state   = merge_dicts(state)
    imu_measurements = merge_dicts(imu_measurements)
    imu_gt = merge_dicts(imu_gt)
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


def traj_end_exit(initial_state, trajectory, using_vio = False):
    """
    Returns a exit function. The exit function returns an exit status message if
    the quadrotor is near hover at the end of the provided trajectory. If the
    initial state is already at the end of the trajectory, the simulation will
    run for at least one second before testing again.
    """

    xf = trajectory.update(np.inf)['x']
    yawf = trajectory.update(np.inf)['yaw']
    rotf = Rotation.from_rotvec(yawf * np.array([0, 0, 1])) # create rotation object that describes yaw
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
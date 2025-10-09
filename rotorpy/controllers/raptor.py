from foundation_policy import Raptor
from scipy.spatial.transform import Rotation
import numpy as np

class FoudationPolicy(object):
    """
    implementing the RAPTOP policy
    """
    def __init__(self, quad_params):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """

        self.policy = Raptor()
        self.policy.reset()

        # initialize the action
        self.action_past = np.zeros(4)

        # load rpm info
        self.rotor_speed_min = quad_params['rotor_speed_min']
        self.rotor_speed_max = quad_params['rotor_speed_max']

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_motor_thrusts, N
                cmd_thrust, N 
                cmd_moment, N*m
                cmd_q, quaternion [i,j,k,w]
                cmd_w, angular rates in the body frame, rad/s
                cmd_v, velocity in the world frame, m/s
        """

        # fetch the state estimates (the policy works under ENU frame)
        pos = state['x']
        vel = state['v']
        R = Rotation.from_quat(state['q']).as_matrix()
        omega = state['w']

        # fetch target pos and vel
        target_pos = flat_output['x']
        target_vel = flat_output['x_dot']
        
        # for trajectory, use pos-target_pos and vel-target_vel
        observation = np.hstack((pos - target_pos, R.flatten(), vel-target_vel, omega, self.action_past))

        # clip policy outputs to be the defined range [-1, 1]
        action = np.clip(self.policy.evaluate_step(observation)[0], -1, 1)
        # save current control action
        self.action_past = action

        # put dummy cmd_moment and cmd_q just to pass the sanity check
        # control-wise, the cmd_motor_speeds has priority
        cmd_moment_dummy = np.zeros((3,))
        cmd_q_dummy = np.array([0,0,0,1])
        cmd_thrust_dummy = 0
        
        # remap the motor numbers
        action_remapped = np.array([action[3], action[0], action[1], action[2]])
        
        # denormalize based on https://github.com/rl-tools/raptor?tab=readme-ov-file#usage
        action_rpm = (self.rotor_speed_max - self.rotor_speed_min) * (action_remapped + 1)/2 + self.rotor_speed_min
        
        # load control
        control_input = {'cmd_motor_speeds':action_rpm,
                         'cmd_thrust':cmd_thrust_dummy,
                         'cmd_moment':cmd_moment_dummy,
                         'cmd_q':cmd_q_dummy}
        
        return control_input
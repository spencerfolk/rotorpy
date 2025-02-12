import numpy as np
import torch
import roma
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """
    Quadrotor trajectory tracking controller based on https://ieeexplore.ieee.org/document/5717652 

    """
    def __init__(self, quad_params):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """

        # Quadrotor physical parameters.
        # Inertial parameters
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.Ixy             = quad_params['Ixy']  # kg*m^2
        self.Ixz             = quad_params['Ixz']  # kg*m^2
        self.Iyz             = quad_params['Iyz']  # kg*m^2

        # Frame parameters
        self.c_Dx            = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.num_rotors      = quad_params['num_rotors']
        self.rotor_pos       = quad_params['rotor_pos']
        self.rotor_dir       = quad_params['rotor_directions']

        # Rotor parameters    
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s

        self.k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
        self.k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
        self.k_d             = quad_params['k_d']       # rotor drag coeff, N/(m/s)
        self.k_z             = quad_params['k_z']       # induced inflow coeff N/(m/s)
        self.k_flap          = quad_params['k_flap']    # Flapping moment coefficient Nm/(m/s)

        # Motor parameters
        self.tau_m           = quad_params['tau_m']     # motor reponse time, seconds

        # You may define any additional constants you like including control gains.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]]) # kg*m^2
        self.g = 9.81 # m/s^2

        # Gains  
        self.kp_pos = np.array([6.5,6.5,15])
        self.kd_pos = np.array([4.0, 4.0, 9])
        self.kp_att = 544
        self.kd_att = 46.64
        self.kp_vel = 0.1*self.kp_pos   # P gain for velocity controller (only used when the control abstraction is cmd_vel)

        # Linear map from individual rotor forces to scalar thrust and vector
        # moment applied to the vehicle.
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient. 

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                  np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), 
                                 (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)
    
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
                cmd_acc, mass normalized thrust vector in the world frame, m/s/s.

                Not all keys are used, it depends on the control_abstraction selected when initializing the Multirotor object. 
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        def normalize(x):
            """Return normalized vector."""
            return x / np.linalg.norm(x)

        def vee_map(S):
            """Return vector corresponding to given skew symmetric matrix."""
            return np.array([-S[1,2], S[0,2], -S[0,1]])

        # Get the desired force vector.
        pos_err  = state['x'] - flat_output['x']
        dpos_err = state['v'] - flat_output['x_dot']
        F_des = self.mass * (- self.kp_pos*pos_err
                             - self.kd_pos*dpos_err
                             + flat_output['x_ddot']
                             + np.array([0, 0, self.g]))

        # Desired thrust is force projects onto b3 axis.
        R = Rotation.from_quat(state['q']).as_matrix()
        b3 = R @ np.array([0, 0, 1])
        u1 = np.dot(F_des, b3)

        # Desired orientation to obtain force vector.
        b3_des = normalize(F_des)
        yaw_des = flat_output['yaw']
        c1_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        b2_des = normalize(np.cross(b3_des, c1_des))
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.stack([b1_des, b2_des, b3_des]).T

        # Orientation error.
        S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
        att_err = vee_map(S_err)

        # Angular velocity error (this is oversimplified).
        w_des = np.array([0, 0, flat_output['yaw_dot']])
        w_err = state['w'] - w_des

        # Desired torque, in units N-m.
        u2 = self.inertia @ (-self.kp_att*att_err - self.kd_att*w_err) + np.cross(state['w'], self.inertia@state['w'])  # Includes compensation for wxJw component

        # Compute command body rates by doing PD on the attitude error. 
        cmd_w = -self.kp_att*att_err - self.kd_att*w_err

        # Compute motor speeds. Avoid taking square root of negative numbers.
        TM = np.array([u1, u2[0], u2[1], u2[2]])
        cmd_rotor_thrusts = self.TM_to_f @ TM
        cmd_motor_speeds = cmd_rotor_thrusts / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        # Assign controller commands.
        cmd_thrust = u1                                             # Commanded thrust, in units N.
        cmd_moment = u2                                             # Commanded moment, in units N-m.
        cmd_q = Rotation.from_matrix(R_des).as_quat()               # Commanded attitude as a quaternion.
        cmd_v = -self.kp_vel*pos_err + flat_output['x_dot']         # Commanded velocity in world frame (if using cmd_vel control abstraction), in units m/s
        cmd_acc = F_des/self.mass                                   # Commanded acceleration in world frame (if using cmd_acc control abstraction)

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_motor_thrusts':cmd_rotor_thrusts,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v,
                         'cmd_acc': cmd_acc}
        
        return control_input


class SE3ControlBatch(object):
    # eventually, we could batch the quad params as well.
    def __init__(self, quad_params, device):
        self.device = device
        # Quadrotor physical parameters
        self.mass = quad_params['mass']
        self.inertia = torch.tensor([[quad_params['Ixx'], quad_params['Ixy'], quad_params['Ixz']],
                                     [quad_params['Ixy'], quad_params['Iyy'], quad_params['Iyz']],
                                     [quad_params['Ixz'], quad_params['Iyz'], quad_params['Izz']]], device=self.device)
        self.g = 9.81

        # Gains
        self.kp_pos = torch.tensor([6.5, 6.5, 15], device=self.device).unsqueeze(0)
        self.kd_pos = torch.tensor([4.0, 4.0, 9], device=self.device).unsqueeze(0)
        self.kp_att = 544
        self.kd_att = 46.64
        self.kp_vel = 0.1 * self.kp_pos

        # Control allocation matrix
        self.k_eta = quad_params['k_eta']
        self.k_m = quad_params['k_m']
        k = self.k_m / self.k_eta
        self.num_rotors = quad_params['num_rotors']
        # self.rotor_pos = torch.tensor(list(quad_params['rotor_pos'].values()))
        # self.rotor_dir = torch.tensor(quad_params['rotor_directions'])

        self.rotor_pos       = quad_params['rotor_pos']
        self.rotor_dir       = quad_params['rotor_directions']


        self.f_to_TM = torch.from_numpy(np.vstack((np.ones((1,self.num_rotors)),
                                                   np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]),
                                                   (k * self.rotor_dir).reshape(1,-1)))).float().to(self.device)
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

    def normalize(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True)

    # TODO(hersh500): I suspect, due to uses of squeeze(), if batch_size==1 then there might be errors.
    def update(self, states, flat_outputs):
        '''
        Computes a batch of control outputs
        :param states: a dictionary of pytorch tensors containing the states of the quadrotors
        :param flat_outputs: a dictionary of pytorch tensors containing the reference trajectories for each quad.
        :return:
        '''
        pos_err = states['x'].float() - flat_outputs['x']
        dpos_err = states['v'].float() - flat_outputs['x_dot']

        F_des = self.mass * (-self.kp_pos * pos_err
                             - self.kd_pos * dpos_err
                             + flat_outputs['x_ddot']
                             + torch.tensor([0, 0, self.g], device=self.device))


        # R = torch.tensor(np.array([Rotation.from_quat(q).as_matrix() for q in states['q']])).float().to(self.device)
        R = roma.unitquat_to_rotmat(states['q']).float()
        b3 = R @ torch.tensor([0.0, 0.0, 1.0], device=self.device).float()
        u1 = torch.sum(F_des * b3, dim=-1).float()

        b3_des = self.normalize(F_des)
        yaw_des = flat_outputs['yaw']
        c1_des = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros_like(yaw_des)], dim=-1)
        b2_des = self.normalize(torch.cross(b3_des, c1_des, dim=-1))
        b1_des = torch.cross(b2_des, b3_des, dim=-1)
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

        S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
        att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

        w_des = torch.stack([torch.zeros_like(yaw_des), torch.zeros_like(yaw_des), flat_outputs['yaw_dot']], dim=-1).to(self.device)
        w_err = states['w'] - w_des

        Iw = self.inertia.unsqueeze(0).float() @ states['w'].unsqueeze(-1)
        x = -self.kp_att * att_err - self.kd_att * w_err
        u2 = (self.inertia.unsqueeze(0).float() @ x.unsqueeze(-1)).squeeze() + torch.cross(states['w'], Iw.squeeze(), dim=-1)

        TM = torch.cat([u1.unsqueeze(-1), u2], dim=-1)
        cmd_rotor_thrusts = self.TM_to_f @ TM.T
        cmd_motor_speeds = cmd_rotor_thrusts / self.k_eta
        cmd_motor_speeds = torch.sign(cmd_motor_speeds) * torch.sqrt(torch.abs(cmd_motor_speeds))

        # cmd_q = torch.tensor([Rotation.from_matrix(r.numpy()).as_quat() for r in R_des], device=self.device)
        cmd_q = roma.rotmat_to_unitquat(R_des)
        cmd_v = -self.kp_vel * pos_err + flat_outputs['x_dot']

        # cmd_motor_speeds_rpm = rad_to_rpm(cmd_motor_speeds)
        control_inputs = {'cmd_motor_speeds': cmd_motor_speeds.T,
                          # 'cmd_motor_speeds_rpm': cmd_motor_speeds_rpm.T,
                          'cmd_thrust': u1,
                          'cmd_moment': u2,
                          'cmd_q': cmd_q,
                          'cmd_w': -self.kp_att * att_err - self.kd_att * w_err,
                          'cmd_v': cmd_v}
        return control_inputs

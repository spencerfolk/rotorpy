import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

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
        # with the z axis and that the "sign" of each rotor yaw moment alternates starting with positive for r1. 'TM' = "thrust and moments"
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                  np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), 
                                 (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

    def update_ref(self, t, flat_output):
        """
        This function receives the current time, and desired flat
        outputs. It returns the reference command inputs.
        Follows https://repository.upenn.edu/edissertations/547/

        Inputs:
            t, present time in seconds
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2  a
                x_dddot,  jerk, m/s**3          a_dot
                x_ddddot, snap, m/s**4          a_ddot
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
                yaw_ddot, yaw acceleration, rad/s**2  #required! not the same if computing command using controller

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
                cmd_w, angular velocity
                cmd_a, angular acceleration
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_q = np.zeros((4,))

        def normalize(x):
            """Return normalized vector."""
            return x / np.linalg.norm(x)

        # Desired force vector.
        t = flat_output['x_ddot']+ np.array([0, 0, self.g])
        b3 = normalize(t) 
        F_des = self.mass * (t)# this is vectorized

        # Control input 1: collective thrust. 
        u1 = np.dot(F_des, b3)

        # Desired orientation to obtain force vector.
        b3_des = normalize(F_des) #b3_des and b3 are the same
        yaw_des = flat_output['yaw']
        c1_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        b2_des = normalize(np.cross(b3_des, c1_des))
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.stack([b1_des, b2_des, b3_des]).T

        R = R_des # assume we have perfect tracking on rotation
        
        # Following section follows Mellinger paper to compute reference angular velocity
        dot_u1 = np.dot(b3,flat_output['x_dddot'])
        hw = self.mass/u1*(flat_output['x_dddot']-dot_u1*b3)
        p  = np.dot(-hw, b2_des)
        q  = np.dot(hw, b1_des)
        w_des = np.array([0, 0, flat_output['yaw_dot']])
        r  = np.dot(w_des, b3_des)
        Omega = np.array([p, q, r])

        wwu1b3 = np.cross(Omega, np.cross(Omega, u1*b3))
        ddot_u1 = np.dot(b3, self.mass*flat_output['x_ddddot']) - np.dot(b3, wwu1b3)
        ha = 1.0/u1*(self.mass*flat_output['x_ddddot'] - ddot_u1*b3 - 2*np.cross(Omega,dot_u1*b3) - wwu1b3)
        p_dot = np.dot(-ha, b2_des)
        q_dot = np.dot(ha, b1_des)
        np.cross(Omega, Omega)
        r_dot = flat_output['yaw_ddot'] *np.dot(np.array([0,0,1.0]), b3_des) #uniquely need yaw_ddot
        Alpha = np.array([p_dot, q_dot, r_dot]) 

        # Control input 2: moment on each body axis
    
        u2 =  self.inertia @ Alpha + np.cross(Omega, self.inertia @ Omega)

        # Convert to cmd motor speeds. 
        TM = np.array([u1, u2[0], u2[1], u2[2]])
        cmd_motor_forces = self.TM_to_f @ TM
        cmd_motor_speeds = cmd_motor_forces / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        cmd_q = Rotation.from_matrix(R_des).as_quat()


        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                        'cmd_thrust':u1,
                        'cmd_moment':u2,
                        'cmd_q':cmd_q,
                        'cmd_w':Omega,
                        'cmd_a':Alpha}
        return control_input
    
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
        cmd_v = -self.kp_vel*pos_err + flat_output['x_dot']     # Commanded velocity in world frame (if using cmd_vel control abstraction), in units m/s

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_motor_thrusts':cmd_rotor_thrusts,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v}
        return control_input
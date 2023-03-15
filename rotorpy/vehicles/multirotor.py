import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.hummingbird_params import quad_params

"""
Multirotor models
"""

def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        duat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[ q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [ q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot

class Multirotor(object):
    """
    Quadrotor forward dynamics model.
    """
    def __init__(self, quad_params, initial_state = {'x': np.array([0,0,0]),
                                            'v': np.zeros(3,),
                                            'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                            'w': np.zeros(3,),
                                            'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                            'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])},
                ):
        """
        Initialize quadrotor physical parameters.
        """

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

        self.extract_geometry()

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
        self.motor_noise     = quad_params['motor_noise_std'] # noise added to the actual motor speed, rad/s / sqrt(Hz)

        # Additional constants.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])
        self.rotor_drag_matrix = np.array([[self.k_d,   0,                 0],
                                           [0,          self.k_d,          0],
                                           [0,          0,          self.k_z]])
        self.drag_matrix = np.array([[self.c_Dx,    0,          0],
                                     [0,            self.c_Dy,  0],
                                     [0,            0,          self.c_Dz]])
        self.g = 9.81 # m/s^2

        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])

        # Set the initial state
        self.initial_state = initial_state

    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of 
        wrenches acting on the rigid body.
        The rotor_geometry is an array of length (n,3), where n is the number of rotors. 
        Each row corresponds to the position vector of the rotor relative to the CoM. 
        """

        self.rotor_geometry = np.array([]).reshape(0,3)
        for rotor in self.rotor_pos:
            r = self.rotor_pos[rotor]
            self.rotor_geometry = np.vstack([self.rotor_geometry, r])

        return

    def statedot(self, state, cmd_rotor_speeds, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds)
        s = Multirotor._pack_state(state)
        
        s_dot = s_dot_fn(0, s)
        v_dot = s_dot[3:6]
        w_dot = s_dot[10:13]

        state_dot = {'vdot': v_dot,'wdot': w_dot}
        return state_dot 


    def step(self, state, cmd_rotor_speeds, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds)
        s = Multirotor._pack_state(state)

        # Option 1 - RK45 integration
        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        s = sol['y'][:,-1]
        # Option 2 - Euler integration
        # s = s + s_dot_fn(0, s) * t_step  # first argument doesn't matter. It's time invariant model

        state = Multirotor._unpack_state(s)

        # Re-normalize unit quaternion.
        state['q'] = state['q'] / norm(state['q'])

        # Add noise to the motor speed measurement
        state['rotor_speeds'] += np.random.normal(scale=np.abs(self.motor_noise), size=(self.num_rotors,))

        return state

    def _s_dot_fn(self, t, s, cmd_rotor_speeds):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = Multirotor._unpack_state(s)

        rotor_speeds = state['rotor_speeds']
        inertial_velocity = state['v']
        wind_velocity = state['wind']

        R = Rotation.from_quat(state['q']).as_matrix()

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.T@(inertial_velocity - wind_velocity)

        # Rotor speed derivative
        rotor_accel = (1/self.tau_m)*(cmd_rotor_speeds - rotor_speeds)

        # Position derivative.
        x_dot = state['v']

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, Mtot) = self.compute_body_wrench(state['w'], rotor_speeds, body_airspeed_vector)

        # Rotate the force from the body frame to the inertial frame
        Ftot = R@FtotB

        # Velocity derivative.
        v_dot = (self.weight + Ftot) / self.mass

        # Angular velocity derivative.
        w = state['w']
        w_hat = Multirotor.hat_map(w)
        w_dot = self.inv_inertia @ (Mtot - w_hat @ (self.inertia @ w))

        # NOTE: the wind dynamics are currently handled in the wind_profile object. 
        # The line below doesn't do anything, as the wind state is assigned elsewhere. 
        wind_dot = np.zeros(3,)

        # Pack into vector of derivatives.
        s_dot = np.zeros((16+self.num_rotors,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot
        s_dot[13:16] = wind_dot
        s_dot[16:]   = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed 
        for aerodynamic forces. 
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame. 
        The net moment Mtot is represented in the body frame. 
        """

        FtotB = np.zeros((3,))
        MtotB = np.zeros((3,))

        for i in range(self.num_rotors):
            # Loop through each rotor, compute the forces 

            r = self.rotor_geometry[i,:] # the position of rotor i relative to the CoM, in body coordinates
            
            # Compute the local airspeed by adding on the rotational component to the airspeed.
            local_airspeed_vector = body_airspeed_vector + Multirotor.hat_map(body_rates)@r

            T = np.array([0, 0, self.k_eta*rotor_speeds[i]**2])             # thrust vector in body frame
            H = -rotor_speeds[i]*self.rotor_drag_matrix@local_airspeed_vector     # rotor drag force

            # Compute the moments
            M_force = Multirotor.hat_map(r)@(T+H)
            M_yaw = self.rotor_dir[i]*np.array([0, 0, self.k_m*rotor_speeds[i]**2]) 
            M_flap = -rotor_speeds[i]*self.k_flap*Multirotor.hat_map(local_airspeed_vector)@np.array([0,0,1])

            FtotB += (T+H)
            MtotB += (M_force + M_yaw + M_flap)

        # Compute the drag force acting on the frame
        D = -Multirotor._norm(body_airspeed_vector)*self.drag_matrix@body_airspeed_vector

        FtotB += D

        return (FtotB, MtotB)

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)    ])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[    0, -s[2],  s[1]],
                         [ s[2],     0, -s[0]],
                         [-s[1],  s[0],     0]])

    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((20,))   # FIXME: this shouldn't be hardcoded. Should vary with the number of rotors. 
        s[0:3]   = state['x']       # inertial position
        s[3:6]   = state['v']       # inertial velocity
        s[6:10]  = state['q']       # orientation
        s[10:13] = state['w']       # body rates
        s[13:16] = state['wind']    # wind vector
        s[16:]   = state['rotor_speeds']     # rotor speeds

        return s

    @classmethod
    def _norm(cls, v):
        """
        Given a vector v in R^3, return the 2 norm (length) of the vector
        """
        norm = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        return norm

    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        """
        state = {'x':s[0:3], 'v':s[3:6], 'q':s[6:10], 'w':s[10:13], 'wind':s[13:16], 'rotor_speeds':s[16:]}
        return state
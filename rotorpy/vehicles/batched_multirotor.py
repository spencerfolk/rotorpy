import numpy as np
from numpy.linalg import inv, norm
import torch
import scipy.integrate
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.hummingbird_params import quad_params
from torchdiffeq import odeint
import time
import roma

"""
Multirotor models, batched using PyTorch
"""
def quat_dot(quat, omega):
    """
    Parameters:
        quat, (...,[i,j,k,w])
        omega, angular velocity of body in body axes: (...,3)

    Returns
        duat_dot, (...,[i,j,k,w])

    """
    b = quat.shape[0]
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[...,0], quat[...,1], quat[...,2], quat[...,3])
    G = torch.stack([q3, q2, -q1, -q0,
                   -q2, q3, q0, -q1,
                   q1, -q0, q3, -q2], dim=1).view((b, 3, 4))

    quat_dot = 0.5 * torch.transpose(G, 1,2) @ omega.unsqueeze(-1)
    # Augment to maintain unit quaternion.
    quat_err = torch.sum(quat**2, dim=-1) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot.squeeze(-1) - quat_err.unsqueeze(-1) * quat_err_grad
    return quat_dot

class BatchedMultirotor(object):
    """
    Batched Multirotor forward dynamics model.

    states: [position, velocity, attitude, body rates, wind, rotor speeds]

    Parameters:
        quad_params: a dictionary containing relevant physical parameters for the multirotor. 
        initial_state: the initial state of the vehicle. 
        control_abstraction: the appropriate control abstraction that is used by the controller, options are...
                                'cmd_motor_speeds': the controller directly commands motor speeds. 
                                'cmd_motor_thrusts': the controller commands forces for each rotor.
                                'cmd_ctbr': the controller commands a collective thrsut and body rates. 
                                'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
                                'cmd_ctatt': the controller commands a collective thrust and attitude (as a quaternion).
                                'cmd_vel': the controller commands a velocity vector in the world frame. 
                                'cmd_acc': the controller commands a mass normalized thrust vector (acceleration) in the world frame.
        aero: boolean, determines whether or not aerodynamic drag forces are computed. 
    """
    def __init__(self, quad_params,
                       num_drones,
                       initial_states,
                       device,
                       control_abstraction='cmd_motor_speeds',
                       aero = True,
                ):
        """
        Initialize quadrotor physical parameters.
        TODO(hersh500): add support for different drone parameters within a batch.
        """

        self.device = device

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

        self.rotor_dir       = torch.from_numpy(quad_params['rotor_directions'])

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
        self.inertia = torch.tensor([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]], device=device).unsqueeze(0).double()  # will be useful to add a batch dim
        self.rotor_drag_matrix = torch.tensor([[self.k_d,   0,                 0],
                                           [0,          self.k_d,          0],
                                           [0,          0,          self.k_z]], device=device).double()
        self.drag_matrix = torch.tensor([[self.c_Dx,    0,          0],
                                     [0,            self.c_Dy,  0],
                                     [0,            0,          self.c_Dz]], device=device).double()
        self.g = 9.81 # m/s^2

        self.inv_inertia = torch.linalg.inv(self.inertia).double()
        self.weight = torch.tensor([0, 0, -self.mass*self.g], device=device)

        # Control allocation
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient. 

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        self.f_to_TM = torch.from_numpy(np.vstack((np.ones((1,self.num_rotors)),
                                                   np.hstack([np.cross(self.rotor_pos[key],
                                                                       np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]),
                                                   (k * self.rotor_dir).reshape(1,-1)))).to(device)
        self.rotor_dir = self.rotor_dir.to(device)
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        # Set the initial state
        self.initial_states = initial_states
        assert(initial_states['x'].device == device, "Initial states must already be on the specified device.")
        assert self.initial_states['x'].shape[0] == num_drones

        self.control_abstraction = control_abstraction

        self.k_w = 1                # The body rate P gain        (for cmd_ctbr)
        self.k_v = 10               # The *world* velocity P gain (for cmd_vel)
        self.kp_att = 544           # The attitude P gain (for cmd_vel, cmd_acc, and cmd_ctatt)
        self.kd_att = 46.64         # The attitude D gain (for cmd_vel, cmd_acc, and cmd_ctatt)

        self.aero = aero
        self.num_drones = num_drones

        # self.compute_body_wrench = torch.jit.trace(self.compute_body_wrench,
        #                                            (torch.rand((self.num_drones, 3)),
        #                                            torch.rand(self.num_drones, 4),
        #                                            torch.rand(self.num_drones, 3)))

    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of 
        wrenches acting on the rigid body.
        The rotor_geometry is an array of length (n,3), where n is the number of rotors. 
        Each row corresponds to the position vector of the rotor relative to the CoM.
        Currently, each drone within a batch must have the same parameters, so we are not iterating over num_drones.
        """

        self.rotor_geometry = np.array([]).reshape(0,3)
        for rotor in self.rotor_pos:
            r = self.rotor_pos[rotor]
            self.rotor_geometry = np.vstack([self.rotor_geometry, r])
        self.rotor_geometry = torch.from_numpy(self.rotor_geometry).unsqueeze(0).to(self.device)  # naive, we could instead build the torch tensor directly.
        return

    def statedot(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds)
        s = BatchedMultirotor._pack_state(state, self.num_drones, self.device)
        
        s_dot = s_dot_fn(0, s)
        v_dot = s_dot[...,3:6]
        w_dot = s_dot[...,10:13]

        state_dot = {'vdot': v_dot,'wdot': w_dot}
        return state_dot 


    # TODO(hersh500): add the ability to selectively step certain drones. This will be useful if some drones
    # crash or finish their trajectories earlier than others.
    def step(self, state, control, t_step, idxs=None, debug=False):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        """
        if idxs is None:
            idxs = [i for i in range(self.num_drones)]
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, debug)
        s = BatchedMultirotor._pack_state(state, self.num_drones, self.device)

        # Option 1 - RK45 integration
        # sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        sol = odeint(s_dot_fn, s, t=torch.tensor([0.0, t_step], device=self.device), method='dopri5')
        # s = sol['y'][:,-1]
        s = sol[-1,:]
        # Option 2 - Euler integration
        # s = s + s_dot_fn(0, s) * t_step  # first argument doesn't matter. It's time invariant model

        state = BatchedMultirotor._unpack_state(s)

        # Re-normalize unit quaternion.
        state['q'] = state['q'] / torch.norm(state['q'], dim=-1).unsqueeze(-1)

        # Add noise to the motor speed measurement
        state['rotor_speeds'] += torch.normal(mean=torch.zeros(self.num_rotors, device=self.device),
                                              std=torch.ones(self.num_rotors, device=self.device) * np.abs(self.motor_noise))
        state['rotor_speeds'] = torch.clip(state['rotor_speeds'], self.rotor_speed_min, self.rotor_speed_max)

        return state

    def _s_dot_fn(self, t, s, cmd_rotor_speeds, debug=False):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = BatchedMultirotor._unpack_state(s)

        rotor_speeds = state['rotor_speeds']
        if debug:
            print(f"batched multirotor rotor speeds: {rotor_speeds}")
        inertial_velocity = state['v']
        wind_velocity = state['wind']

        # R = Rotation.from_quat(state['q']).as_matrix()
        R = roma.unitquat_to_rotmat(state['q']).double()

        # Rotor speed derivative
        rotor_accel = (1/self.tau_m)*(cmd_rotor_speeds - rotor_speeds)
        if debug:
            print(f"multirotor s_dot_fn: rotor_accel: {rotor_accel}")

        # Position derivative.
        x_dot = state['v']

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.transpose(1, 2)@(inertial_velocity - wind_velocity).unsqueeze(-1).double()

        body_airspeed_vector = body_airspeed_vector.squeeze(-1)
        if debug:
            print(f"multirotor s_dot_fn: q_dot: {q_dot}")  # INCORRECT
            print(f"multirotor s_dot_fn: body_airspeed_vector: {body_airspeed_vector}")  # CORRECT

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'], rotor_speeds, body_airspeed_vector)

        # Rotate the force from the body frame to the inertial frame
        Ftot = R@FtotB.unsqueeze(-1)

        # Velocity derivative.
        v_dot = (self.weight + Ftot.squeeze(-1)) / self.mass

        # Angular velocity derivative.
        w = state['w'].double()
        w_hat = BatchedMultirotor.hat_map(w).permute(2, 0, 1)
        w_dot = self.inv_inertia @ (MtotB - (w_hat.double() @ (self.inertia @ w.unsqueeze(-1))).squeeze(-1)).unsqueeze(-1)
        if debug:
            print(f"batched multirotor FtotB: {FtotB}") # CORRECT
            print(f"batched multirotor w_dot: {w_dot}") # CORRECT
            print(f"batched multirotor v_dot: {v_dot}") # CORRECT

        # NOTE: the wind dynamics are currently handled in the wind_profile object. 
        # The line below doesn't do anything, as the wind state is assigned elsewhere. 
        wind_dot = torch.zeros((self.num_drones, 3), device=self.device)

        # Pack into vector of derivatives.
        s_dot = torch.zeros((self.num_drones, 16+self.num_rotors,), device=self.device)
        s_dot[:,0:3]   = x_dot
        s_dot[:,3:6]   = v_dot
        s_dot[:,6:10]  = q_dot
        s_dot[:,10:13] = w_dot.squeeze(-1)
        s_dot[:,13:16] = wind_dot
        s_dot[:,16:]   = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector, debug=False):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed 
        for aerodynamic forces. 
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame. 
        The net moment Mtot is represented in the body frame. 
        """

        # Get the local airspeeds for each rotor
        local_airspeeds = body_airspeed_vector.unsqueeze(-1) + (BatchedMultirotor.hat_map(body_rates).permute(2, 0, 1))@(self.rotor_geometry.transpose(1,2))

        # Compute the thrust of each rotor, assuming that the rotors all point in the body z direction!
        T = torch.zeros(self.num_drones, 3, 4, device=self.device)
        T[...,-1,:] = self.k_eta * rotor_speeds**2

        # Add in aero wrenches (if applicable)
        if self.aero:
            # Parasitic drag force acting at the CoM
            tmp = self.drag_matrix.unsqueeze(0)@(body_airspeed_vector).unsqueeze(-1)
            D = -BatchedMultirotor._norm(body_airspeed_vector).unsqueeze(-1)*tmp.squeeze()
            # Rotor drag (aka H force) acting at each propeller hub.
            tmp = self.rotor_drag_matrix.unsqueeze(0)@local_airspeeds.double()
            H = -rotor_speeds.unsqueeze(1)*tmp
            # Pitching flapping moment acting at each propeller hub.
            M_flap = BatchedMultirotor.hat_map(local_airspeeds.transpose(1, 2).reshape(self.num_drones*4, 3))
            M_flap = M_flap.permute(2, 0, 1).reshape(self.num_drones, 4, 3, 3).double()
            M_flap = M_flap@torch.tensor([0,0,1.0]).double()
            M_flap = -self.k_flap*rotor_speeds.unsqueeze(1)*M_flap.transpose(-1, -2)
        else:
            D = torch.zeros(self.num_drones, 3, device=self.device)
            H = torch.zeros((self.num_drones, 3, self.num_rotors), device=self.device)
            M_flap = torch.zeros((self.num_drones, 3,self.num_rotors), device=self.device)

        # Compute the moments due to the rotor thrusts, rotor drag (if applicable), and rotor drag torques
        # install opt-einsum https://pytorch.org/docs/stable/generated/torch.einsum.html
        M_force = -torch.einsum('bijk, bik->bj', BatchedMultirotor.hat_map(self.rotor_geometry.squeeze()).unsqueeze(0).double(), T+H)
        M_yaw = torch.zeros(self.num_drones, 3, 4, device=self.device)
        M_yaw[...,-1,:] = self.rotor_dir * self.k_m * rotor_speeds**2

        if debug:
            print(f"batched multirotor compute_body_wrench: local_airspeeds: {local_airspeeds}")  # CORRECT
            print(f"batched multirotor compute_body_wrench: T: {T}")  # CORRECT
            print(f"batched multirotor compute_body_wrench: D: {D}")  # CORRECT
            print(f"batched multirotor compute_body_wrench: H: {H}")  # CORRCT
            print(f"batched multirotor compute_body_wrench: M_flap: {M_flap}")
            print(f"batched multirotor compute_body_wrench: M_force: {M_force}")  # CORRECT
            print(f"batched multirotor compute_body_wrench: M_yaw: {M_yaw}")  # CORRECT

        # Sum all elements to compute the total body wrench
        FtotB = torch.sum(T + H, dim=2) + D
        MtotB = M_force + torch.sum(M_yaw + M_flap, dim=2)

        return (FtotB, MtotB)

    def get_cmd_motor_speeds(self, state, control):
        """
        Computes the commanded motor speeds depending on the control abstraction.
        For higher level control abstractions, we have low-level controllers that will produce motor speeds based on the higher level commmand. 

        """

        if self.control_abstraction == 'cmd_motor_speeds':
            # The controller directly controls motor speeds, so command that. 
            return control['cmd_motor_speeds']
        else:
            raise ValueError("Invalid control abstraction selected for BatchedMultirotor. Options are: cmd_motor_speeds")


    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)    ])


    # TODO(hersh500): for torch.jit.trace, this needs to be implemented fully in torch.
    # TODO(hersh500): this will be slow on gpu, due to numpy -> gpu data transfer.
    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
        """
        device = s.device
        if len(s.shape) > 1:  # Vectorized implementation
            s = s.cpu()
            return torch.from_numpy(np.array([[ np.zeros(s.shape[0]), -s[:,2],  s[:,1]],
                             [ s[:,2],     np.zeros(s.shape[0]), -s[:,0]],
                             [-s[:,1],  s[:,0],     np.zeros(s.shape[0])]])).to(device)
            # This is extremely slow/incorrect???
            # s = s.unsqueeze(-1)
            # hat = torch.cat([torch.zeros(s.shape[0], 1, device=device), -s[:, 2], s[:,1],
            #                  s[:,2], torch.zeros(s.shape[0], 1, device=device), -s[:,0],
            #                  -s[:,1], s[:,0], torch.zeros(s.shape[0], 1, device=device)], dim=1).view(3, 3, s.shape[0]).float()
            # return hat
        else:
            # return torch.from_numpy(np.array([[    0, -s[2],  s[1]],
            #                  [ s[2],     0, -s[0]],
            #                  [-s[1],  s[0],     0]]))
            return torch.tensor([[0, -s[2], s[1]],
                                [s[2], 0, -s[0]],
                                 [-s[1], s[0], 0]], device=device)

    @classmethod
    def _pack_state(cls, state, num_drones, device):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = torch.zeros(num_drones, 20, device=device).double()   # FIXME: this shouldn't be hardcoded. Should vary with the number of rotors.
        s[...,0:3]   = state['x']       # inertial position
        s[...,3:6]   = state['v']       # inertial velocity
        s[...,6:10]  = state['q']       # orientation
        s[...,10:13] = state['w']       # body rates
        s[...,13:16] = state['wind']    # wind vector
        s[...,16:]   = state['rotor_speeds']     # rotor speeds

        return s

    @classmethod
    def _norm(cls, v):
        """
        Given a vector v in R^3, return the 2 norm (length) of the vector
        """
        # norm = (v[...,0]**2 + v[...,1]**2 + v[...,2]**2)**0.5
        norm = torch.linalg.norm(v, dim=-1)
        return norm

    # TODO(hersh500): make this work with selected indexes.
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
        # fill state with zeros, then replace with appropriate indexes.
        state = {'x':s[...,0:3], 'v':s[...,3:6], 'q':s[...,6:10], 'w':s[...,10:13], 'wind':s[...,13:16], 'rotor_speeds':s[...,16:]}
        return state


def merge_rotorpy_states(states):
    array_keys = ["x", "v", "q", "w", "wind", "rotor_speeds"]
    merged_states = {}
    for key in array_keys:
        merged_states[key] = torch.cat([torch.from_numpy(f[key]).unsqueeze(0).double() for f in states], dim=0)
    return merged_states


def merge_flat_outputs(flat_outputs, device):
    # these are the important keys that the controller uses
    array_keys = ["x", "x_dot", 'x_ddot']
    scalar_keys = ["yaw", "yaw_dot"]
    merged_flat_outputs = {}
    for key in array_keys:
        merged_flat_outputs[key] = torch.cat([torch.from_numpy(f[key]).unsqueeze(0).double() for f in flat_outputs], dim=0).double().to(device)
    for key in scalar_keys:
        merged_flat_outputs[key] = torch.from_numpy(np.array([f[key] for f in flat_outputs])).double().to(device)
    return merged_flat_outputs

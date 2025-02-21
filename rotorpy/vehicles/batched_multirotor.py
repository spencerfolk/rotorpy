import numpy as np
import torch
from torchdiffeq import odeint
import roma
from typing import List

class BatchedDynamicsParams:
    def __init__(self, quad_params_list, num_drones, device):
        assert len(quad_params_list) == num_drones
        self.num_drones = num_drones
        self.device = device
        self.mass = torch.tensor([quad_params['mass'] for quad_params in quad_params_list]).unsqueeze(-1).to(device) # kg

        self.num_rotors = quad_params_list[0]["num_rotors"]
        for quad_params in quad_params_list:
            assert quad_params["num_rotors"] == self.num_rotors

        self.rotor_pos = [quad_params["rotor_pos"] for quad_params in quad_params_list]

        self.rotor_dir       = np.array([qp['rotor_directions'] for qp in quad_params_list])

        self.extract_geometry()

        # Rotor parameters
        self.rotor_speed_min = torch.tensor([quad_params['rotor_speed_min'] for quad_params in quad_params_list]).unsqueeze(-1).to(device) # rad/s
        self.rotor_speed_max = torch.tensor([quad_params['rotor_speed_max'] for quad_params in quad_params_list]).unsqueeze(-1).to(device) # rad/s

        self.k_eta           = np.array([quad_params['k_eta'] for quad_params in quad_params_list])  # .unsqueeze(-1).to(device)     # thrust coeff, N/(rad/s)**2
        self.k_m           = np.array([quad_params['k_m'] for quad_params in quad_params_list])  # .unsqueeze(-1).to(device)
        self.k_flap           = torch.tensor([quad_params['k_flap'] for quad_params in quad_params_list]).unsqueeze(-1).to(device)   # Flapping moment coefficient Nm/(m/s)

        # Motor parameters
        self.tau_m           = torch.tensor([quad_params['tau_m'] for quad_params in quad_params_list], device=device).unsqueeze(-1)
        self.motor_noise     = torch.tensor([quad_params['motor_noise_std'] for quad_params in quad_params_list], device=device).unsqueeze(-1) # noise added to the actual motor speed, rad/s / sqrt(Hz)

        # Additional constants.
        self.inertia = torch.from_numpy(np.array([[[qp["Ixx"], qp["Ixy"], qp["Ixz"]],
                                                   [qp["Ixy"], qp["Iyy"], qp["Iyz"]],
                                                   [qp["Ixz"], qp["Iyz"], qp["Izz"]]] for qp in quad_params_list])).double().to(device)
        self.rotor_drag_matrix = torch.tensor([[[qp["k_d"],   0,                 0],
                                                [0,          qp["k_d"],          0],
                                                [0,          0,          qp["k_z"]]] for qp in quad_params_list], device=device).double()

        self.drag_matrix = torch.tensor([[[qp["c_Dx"],   0,                 0],
                                          [0,          qp["c_Dy"],          0],
                                          [0,          0,          qp["c_Dz"]]] for qp in quad_params_list], device=device).double()

        # keep this the same across drones.
        self.g = 9.81 # m/s^2

        self.inv_inertia = torch.linalg.inv(self.inertia).double()
        self.weight = torch.zeros(num_drones, 3, device=device).double()
        self.weight[:,-1] = -self.mass.squeeze(-1) * self.g

        # Control allocation
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient.

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        # This will be slow.
        self.f_to_TM = torch.stack([torch.from_numpy(np.vstack((np.ones((1,self.num_rotors)),
                                                                np.hstack([np.cross(self.rotor_pos[i][key],
                                                                                    np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos[i]]),
                                                                (k[i] * self.rotor_dir[i]).reshape(1,-1)))).to(device) for i in range(num_drones)])
        self.k_eta = torch.from_numpy(self.k_eta).unsqueeze(-1).to(device)
        self.k_m = torch.from_numpy(self.k_m).unsqueeze(-1).to(device)
        self.rotor_dir = torch.from_numpy(self.rotor_dir).to(device)
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)


    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of
        wrenches acting on the rigid body.
        The rotor_geometry is an array of length (n,3), where n is the number of rotors.
        Each row corresponds to the position vector of the rotor relative to the CoM.
        Currently, each drone within a batch must have the same parameters, so we are not iterating over num_drones.
        """

        geoms = []
        geom_hat_maps = []
        for i in range(self.num_drones):
            rotor_geometry = np.array([]).reshape(0, 3)
            for rotor in self.rotor_pos[i]:
                r = self.rotor_pos[i][rotor]
                rotor_geometry = np.vstack([rotor_geometry, r])
            geoms.append(rotor_geometry)
            geom_hat_maps.append(BatchedMultirotor.hat_map(torch.from_numpy(rotor_geometry.squeeze())).numpy())
        self.rotor_geometry = torch.from_numpy(np.array(geoms)).to(self.device)
        self.rotor_geometry_hat_maps = torch.from_numpy(np.array(geom_hat_maps)).to(self.device)


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
        integrator: str, "dopri5" or "rk4", which are adaptive or fixed step size integrators. "rk4" will be faster, but potentially less accurate.
    """
    def __init__(self, batched_params: BatchedDynamicsParams,
                       num_drones,
                       initial_states,
                       device,
                       control_abstraction='cmd_motor_speeds',
                       aero = True,
                       integrator='dopri5'
                ):
        """
        Initialize quadrotor physical parameters.
        """
        assert initial_states['x'].device == device, "Initial states must already be on the specified device."
        assert initial_states['x'].shape[0] == num_drones
        assert batched_params.device == device
        self.num_drones = num_drones
        self.device = device
        self.params = batched_params

        # Set the initial state
        self.initial_states = initial_states
        self.control_abstraction = control_abstraction

        # For now, keep these as scalars. But maybe later, we can parametrize them as well.
        self.k_w = 1                # The body rate P gain        (for cmd_ctbr)
        self.k_v = 10               # The *world* velocity P gain (for cmd_vel)
        self.kp_att = 544           # The attitude P gain (for cmd_vel, cmd_acc, and cmd_ctatt)
        self.kd_att = 46.64         # The attitude D gain (for cmd_vel, cmd_acc, and cmd_ctatt)

        self.aero = aero

        assert integrator == 'dopri5' or integrator == "rk4"
        self.integrator = integrator



    def statedot(self, state, control, t_step, idxs):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds, self.params.rotor_speed_min, self.params.rotor_speed_max)

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, idxs)
        s = BatchedMultirotor._pack_state(state, self.num_drones, self.device)
        
        s_dot = s_dot_fn(0, s)
        v_dot = s_dot[...,3:6]
        w_dot = s_dot[...,10:13]

        state_dot = {'vdot': v_dot,'wdot': w_dot}
        return state_dot 


    def step(self, state, control, t_step, idxs=None):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        idxs: integer array of shape (num_running_drones, )
        """
        if idxs is None:
            idxs = [i for i in range(self.num_drones)]
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds, self.params.rotor_speed_min, self.params.rotor_speed_max)

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, idxs)
        s = BatchedMultirotor._pack_state(state, self.num_drones, self.device)

        # Option 1 - RK45 integration
        # sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        sol = odeint(s_dot_fn, s[idxs], t=torch.tensor([0.0, t_step], device=self.device), method=self.integrator)
        # s = sol['y'][:,-1]
        s = sol[-1,:]
        # Option 2 - Euler integration
        # s = s + s_dot_fn(0, s) * t_step  # first argument doesn't matter. It's time invariant model

        state = BatchedMultirotor._unpack_state(s, idxs, self.num_drones)

        # Re-normalize unit quaternion.
        state['q'][idxs] = state['q'][idxs] / torch.norm(state['q'][idxs], dim=-1).unsqueeze(-1)

        # Add noise to the motor speed measurement
        state['rotor_speeds'][idxs] += torch.normal(mean=torch.zeros(self.params.num_rotors, device=self.device),
                                              std=torch.ones(len(idxs), self.params.num_rotors, device=self.device) * torch.abs(self.params.motor_noise[idxs]))
        state['rotor_speeds'][idxs] = torch.clip(state['rotor_speeds'][idxs], self.params.rotor_speed_min[idxs], self.params.rotor_speed_max[idxs])

        return state

    def _s_dot_fn(self, t, s, cmd_rotor_speeds, idxs):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        # so this will be zero for some stuff.
        state = BatchedMultirotor._unpack_state(s, idxs, self.num_drones)

        rotor_speeds = state['rotor_speeds'][idxs]
        inertial_velocity = state['v'][idxs]
        wind_velocity = state['wind'][idxs]

        # R = Rotation.from_quat(state['q']).as_matrix()
        R = roma.unitquat_to_rotmat(state['q'][idxs]).double()

        # Rotor speed derivative
        rotor_accel = (1/self.params.tau_m[idxs])*(cmd_rotor_speeds[idxs] - rotor_speeds)

        # Position derivative.
        x_dot = state['v'][idxs]

        # Orientation derivative.
        q_dot = quat_dot(state['q'][idxs], state['w'][idxs])

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.transpose(1, 2)@(inertial_velocity - wind_velocity).unsqueeze(-1).double()
        body_airspeed_vector = body_airspeed_vector.squeeze(-1)

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'][idxs], rotor_speeds, body_airspeed_vector, idxs)

        # Rotate the force from the body frame to the inertial frame
        Ftot = R@FtotB.unsqueeze(-1)

        # Velocity derivative.
        v_dot = (self.params.weight[idxs] + Ftot.squeeze(-1)) / self.params.mass[idxs]

        # Angular velocity derivative.
        w = state['w'][idxs].double()
        w_hat = BatchedMultirotor.hat_map(w).permute(2, 0, 1)
        w_dot = self.params.inv_inertia[idxs] @ (MtotB - (w_hat.double() @ (self.params.inertia[idxs] @ w.unsqueeze(-1))).squeeze(-1)).unsqueeze(-1)

        # NOTE: the wind dynamics are currently handled in the wind_profile object. 
        # The line below doesn't do anything, as the wind state is assigned elsewhere. 
        wind_dot = torch.zeros((len(idxs), 3), device=self.device)

        # Pack into vector of derivatives.
        s_dot = torch.zeros((len(idxs), 16+self.params.num_rotors,), device=self.device)
        s_dot[:,0:3]   = x_dot
        s_dot[:,3:6]   = v_dot
        s_dot[:,6:10]  = q_dot
        s_dot[:,10:13] = w_dot.squeeze(-1)
        s_dot[:,13:16] = wind_dot
        s_dot[:,16:]   = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector, idxs):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed 
        for aerodynamic forces. 
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame.
        The net moment Mtot is represented in the body frame. 
        """
        assert(body_rates.shape[0] == rotor_speeds.shape[0])
        assert(body_rates.shape[0] == body_airspeed_vector.shape[0])

        num_drones = body_rates.shape[0]
        # Get the local airspeeds for each rotor
        local_airspeeds = body_airspeed_vector.unsqueeze(-1) + (BatchedMultirotor.hat_map(body_rates).permute(2, 0, 1))@(self.params.rotor_geometry[idxs].transpose(1,2))

        # Compute the thrust of each rotor, assuming that the rotors all point in the body z direction!
        T = torch.zeros(num_drones, 3, 4, device=self.device)
        T[...,-1,:] = self.params.k_eta[idxs] * rotor_speeds**2

        # Add in aero wrenches (if applicable)
        if self.aero:
            # Parasitic drag force acting at the CoM
            tmp = self.params.drag_matrix[idxs]@(body_airspeed_vector).unsqueeze(-1)
            D = -BatchedMultirotor._norm(body_airspeed_vector).unsqueeze(-1)*tmp.squeeze()
            # Rotor drag (aka H force) acting at each propeller hub.
            tmp = self.params.rotor_drag_matrix[idxs]@local_airspeeds.double()
            H = -rotor_speeds.unsqueeze(1)*tmp
            # Pitching flapping moment acting at each propeller hub.
            M_flap = BatchedMultirotor.hat_map(local_airspeeds.transpose(1, 2).reshape(num_drones*4, 3))
            M_flap = M_flap.permute(2, 0, 1).reshape(num_drones, 4, 3, 3).double()
            M_flap = M_flap@torch.tensor([0,0,1.0], device=self.device).double()
            # print(f"k flap shape is {self.k_flap.shape}")
            # print(f"rotor speeds shape is {rotor_speeds.shape}")
            # print(f"M_flap shape is is {M_flap.shape}")
            # print(f"intermediate shape is is {(-self.k_flap[idxs]*rotor_speeds).shape}")
            # print(f"M_flap transpose shape  is {M_flap.transpose(-1, -2).shape}")
            M_flap = (-self.params.k_flap[idxs]*rotor_speeds).unsqueeze(1)*M_flap.transpose(-1, -2)
        else:
            D = torch.zeros(num_drones, 3, device=self.device)
            H = torch.zeros((num_drones, 3, self.params.num_rotors), device=self.device)
            M_flap = torch.zeros((num_drones, 3,self.params.num_rotors), device=self.device)

        # Compute the moments due to the rotor thrusts, rotor drag (if applicable), and rotor drag torques
        # install opt-einsum https://pytorch.org/docs/stable/generated/torch.einsum.html
        # M_force = -torch.einsum('bijk, bik->bj', BatchedMultirotor.hat_map(self.rotor_geometry.squeeze()).unsqueeze(0).double(), T+H)

        M_force = -torch.einsum('bijk, bik->bj', self.params.rotor_geometry_hat_maps[idxs], T+H)
        M_yaw = torch.zeros(num_drones, 3, 4, device=self.device)
        M_yaw[...,-1,:] = self.params.rotor_dir[idxs] * self.params.k_m[idxs] * rotor_speeds**2

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


    # Slower on GPU until N >= 4e5
    # Won't work on numpy < 2.2.2 or torch < 2, I think!
    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
        """
        device = s.device
        if len(s.shape) > 1:  # Vectorized implementation
            s = s.unsqueeze(-1)
            hat = torch.cat([torch.zeros(s.shape[0], 1, device=device), -s[:, 2], s[:,1],
                             s[:,2], torch.zeros(s.shape[0], 1, device=device), -s[:,0],
                             -s[:,1], s[:,0], torch.zeros(s.shape[0], 1, device=device)], dim=0).view(3, 3, s.shape[0]).double()
            return hat
        else:
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

    @classmethod
    def _unpack_state(cls, s, idxs, num_drones):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        """
        device = s.device
        state = {'x': torch.full((num_drones, 3), float("nan"), device=device).double(),
                 'v': torch.full((num_drones, 3), float("nan"), device=device).double(),
                 'q': torch.full((num_drones, 4), float("nan"), device=device).double(),
                 'w': torch.full((num_drones, 3), float("nan"), device=device).double(),
                 'wind': torch.full((num_drones, 3), float("nan"), device=device).double(),
                 'rotor_speeds': torch.full((num_drones, 4), float("nan"), device=device).double()}
        state['q'][...,-1] = 1  # make sure we're returning a valid quaternion
        state['x'][idxs] = s[:,0:3]
        state['v'][idxs] = s[:,3:6]
        state['q'][idxs] = s[:,6:10]
        state['w'][idxs] = s[:,10:13]
        state['wind'][idxs] = s[:,13:16]
        state['rotor_speeds'][idxs] = s[:,16:]
        return state

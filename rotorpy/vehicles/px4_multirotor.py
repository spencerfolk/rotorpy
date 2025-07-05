from rotorpy.vehicles.multirotor import Multirotor
from pymavlink import mavutil
import numpy as np
import types

# 10040_sihsim_quadx
sihsim_quadx = {
    'mass':             1.0,     # kg      (PX4 param SIH_MASS)
    'Ixx':              0.025,   # kg·m²   (PX4 param SIH_IXX)
    'Iyy':              0.025,   # kg·m²   (PX4 param SIH_IYY)
    'Izz':              0.030,   # kg·m²   (PX4 param SIH_IZZ)
    'Ixy':              0.0,
    'Ixz':              0.0,
    'Iyz':              0.0,

    'num_rotors':       4,       # CA_ROTOR_COUNT
    'rotor_pos': {
        'r1':           np.array([ 1.0,  1.0, 0.0]),  # CA_ROTOR0_PX/PY
        'r2':           np.array([-1.0, -1.0, 0.0]),  # CA_ROTOR1_PX/PY
        'r3':           np.array([ 1.0, -1.0, 0.0]),  # CA_ROTOR2_PX/PY
        'r4':           np.array([-1.0,  1.0, 0.0]),  # CA_ROTOR3_PX/PY
    },

    # rotor_directions: needs the sign for each motor’s moment (from CA_ROTORn_KM)
    # CA_ROTOR0_KM = +0.05 → +1
    # CA_ROTOR1_KM = +0.05 → +1
    # CA_ROTOR2_KM = -0.05 → -1
    # CA_ROTOR3_KM = -0.05 → -1
    'rotor_directions': np.array([ 1, 1, -1, -1 ]),
    'rI':               np.array([0.0, 0.0, 0.0]),

    'c_Dx':             1.0,
    'c_Dy':             1.0,
    'c_Dz':             1.0,

    'k_eta':            5.0,    # max thrust per rotor (N) → SIH_T_MAX
    'k_m':              0.1,    # max yaw moment per rotor (Nm) → SIH_Q_MAX
    'k_d':              0.0,    # rotor drag
    'k_z':              0.0,    # induced inflow
    'k_h':              0.0,    # translational lift
    'k_flap':           0.0,    # blade flapping moment

    'tau_m':            0.05,   # Motor response time constant (s) ← SIH_T_TAU
    'rotor_speed_min':  0.0,    # zero throttle
    'rotor_speed_max':  1.0,    # full throttle
    'motor_noise_std':  0.0,    # SIH doesn't inject noise by default
}

def compute_hover_state(mass, k_eta, num_rotors, g=9.81):
    """
    Solve N·k_eta·ω² = m·g for ω and return an array of length num_rotors.
    """
    omega = np.sqrt((mass * g) / (num_rotors * k_eta))
    return np.full(num_rotors, omega)

hover_state = {
    'x': np.zeros(3),
    'v': np.zeros(3),
    'q': np.array([0, 0, 0, 1]),
    'w': np.zeros(3),
    'wind': np.zeros(3),
    'rotor_speeds': compute_hover_state(
                        sihsim_quadx['mass'],
                        sihsim_quadx['k_eta'],
                        sihsim_quadx['num_rotors']
                    )
}

initial_state = {
    'x':            np.zeros(3),   # position (m)
    'v':            np.zeros(3),   # velocity (m/s)
    'q':            np.zeros(4),   # quaternion [i, j, k, w] – all zeros here
    'w':            np.zeros(3),   # body rates (rad/s)
    'wind':         np.zeros(3),   # no wind
    'rotor_speeds': np.zeros(
                        sihsim_quadx['num_rotors']
                    )               # all rotors stopped
}

class PX4Multirotor(Multirotor):
    def __init__(
            self,
            quad_params=sihsim_quadx,
            initial_state=hover_state,
            control_abstraction="cmd_motor_speeds",
            aero=True,
            enable_ground=False,
            mavlink_url="udp:127.0.0.1:14540"
    ):
        super().__init__(
            quad_params=quad_params,
            initial_state=initial_state,
            control_abstraction=control_abstraction,
            aero=aero,
            enable_ground=enable_ground,
        )
        self.sensor_data = types.SimpleNamespace(
            accel           = np.zeros(3),
            gyro            = np.zeros(3),
            mag             = np.zeros(3),
            abs_pressure    = 0.0,
            diff_pressure   = 0.0,
            pressure_alt    = 0.0,
            temperature     = 0.0,
        )
        self.t = 0.0
        self.conn = mavutil.mavlink_connection(mavlink_url)
        self.conn.wait_heartbeat()

    def step(self, state, control, t_step):
        msg = self.conn.recv_match(type='HIL_ACTUATOR_CONTROLS',
                                   blocking=False, timeout=0.0)
        if (msg):
            control = {
                'cmd_motor_speeds': list(msg.controls[:self.num_rotors])
            }

        state = super().step(state, control, t_step)
        self.state = state
        self.t += t_step

        ts = int(self.t * 1e6)
        sd = self.sensor_data
        # q = tuple(state['q'])
        # w = tuple(state['w'])
        # x = tuple(state['x'])
        # v = tuple(state['v'])
        #
        # self.conn.mav.hil_state_quaternion_send(
        #     ts,
        #     *q,
        #     *w,
        #     *x,
        #     *v,
        #     0.0,
        #     0.0
        # )

        self.conn.mav.hil_sensor_send(
            ts,
            *tuple(sd.accel),
            *tuple(sd.gyro),
            *tuple(sd.mag),
            sd.abs_pressure,
            sd.diff_pressure,
            sd.pressure_alt,
            sd.temperature,
            fields_updated=0xFFFFFFFF
        )

        return state


from rotorpy.vehicles.multirotor import Multirotor
from pymavlink import mavutil
import numpy as np
import types
from pymavlink.dialects.v20.ardupilotmega import MAVLink

# Constants
R_EARTH = 6378137.0  # meters
rad2deg = 180.0 / np.pi
INT_MAX = 32767
INT_MIN = -32768

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
            mavlink_url="tcpin:localhost:4560"
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
        print("[DEBUG]: PX4Multirotor: MAVLink connection established.")

        # Try to capture an initial geodetic reference from PX4 (lat/lon in degE7, alt in mm)
        self.lat0_e7 = None
        self.lon0_e7 = None
        self.alt0_mm = None

        # Prefer HOME_POSITION or GLOBAL_POSITION_INT if available quickly
        self.conn.mav.request_data_stream_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,  # 10 Hz request (best-effort)
            1
        )
        # Poll a few times non-blocking to obtain an initial fix
        for _ in range(50):  # ~ a few hundred ms in total across loop iterations
            m = self.conn.recv_match(blocking=False, timeout=0.0)
            if not m:
                continue
            msg_type = m.get_type()
            if msg_type == 'HOME_POSITION':
                # HOME_POSITION.alt is in mm; lat/lon are degE7
                self.lat0_e7 = int(m.latitude)
                self.lon0_e7 = int(m.longitude)
                self.alt0_mm = int(m.altitude)
                break
            if msg_type == 'GLOBAL_POSITION_INT':
                self.lat0_e7 = int(m.lat)
                self.lon0_e7 = int(m.lon)
                self.alt0_mm = int(m.alt)
                break
        # Fallback if nothing arrived yet
        if self.lat0_e7 is None:
            self.lat0_e7 = 0
            self.lon0_e7 = 0
            self.alt0_mm = 0
            print("WARNING: PX4Multirotor could not obtain initial position reference from PX4"
                  "\nUsing defaults [0 lat, 0 lon, 0 mm alt].")

    def _local_enu_to_geodetic(self, x_enu):
        """
        Convert local ENU position (meters) to (lat_e7, lon_e7, alt_mm)
        using a flat-earth tangent plane approximation around (lat0, lon0, alt0).
        Assumes x_enu = [x_east, y_north, z_up].
        """
        # If we don't have a reference, just return zeros
        lat0_e7 = self.lat0_e7
        lon0_e7 = self.lon0_e7
        alt0_mm = self.alt0_mm
        if lat0_e7 == 0 and lon0_e7 == 0 and alt0_mm == 0:
            return 0, 0, int(x_enu[2] * 1000.0)  # best-effort: treat local z as AMSL delta

        # Local ENU displacements
        east = float(x_enu[0])
        north = float(x_enu[1])
        up = float(x_enu[2])

        lat0_deg = lat0_e7 / 1e7
        lon0_deg = lon0_e7 / 1e7
        lat0_rad = np.deg2rad(lat0_deg)

        dlat_deg = (north / R_EARTH) * rad2deg
        # Guard cos(lat) near the poles
        cos_lat = np.cos(lat0_rad)
        if abs(cos_lat) < 1e-6:
            cos_lat = np.sign(cos_lat) * 1e-6 if cos_lat != 0.0 else 1e-6
        dlon_deg = (east / (R_EARTH * cos_lat)) * rad2deg

        lat_e7 = int(np.round((lat0_deg + dlat_deg) * 1e7))
        lon_e7 = int(np.round((lon0_deg + dlon_deg) * 1e7))
        alt_mm = int(np.round(alt0_mm + up * 1000.0))
        return lat_e7, lon_e7, alt_mm

    def simulate_magnetic_field(self, q, noise_std=0.05):
        """
        Generate a simulated magnetic field vector (NED, gauss) and rotate it into the body frame using the drone's orientation quaternion.
        q: quaternion [i, j, k, w] (PX4 convention)
        Returns: mag_body (3,) array in gauss
        """
        # Nominal Earth field (NED, μT): North, East, Down
        # Example: 20μT north, 0μT east, 40μT down (inclination ~63°)
        earth_field_ned = np.array([20.0, 0.0, 40.0])

        # PX4 quaternion is [i, j, k, w]
        x, y, z, w = q

        # Rotation matrix from NED to body (quaternion to DCM)
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
        ])

        # Rotate NED field into body frame (still in μT)
        mag_body = R @ earth_field_ned

        # Add Gaussian noise (μT)
        mag_body += np.random.normal(0, noise_std, size=3)

        # Convert μT to gauss (1 gauss = 100 μT)
        mag_body_gauss = mag_body / 100.0

        # Store in sensor_data
        self.sensor_data.mag = mag_body_gauss
        return mag_body_gauss

    def simulate_pressure(self, z_up_m):
        """
        Simulate barometric pressure (Pa) at altitude z_up_m (meters above sea level, z=0 at sea level, z up positive).
        Uses a simple exponential model: P = P0 * exp(-z/H)
        P0: sea level standard pressure (101325 Pa)
        H: scale height (~8434 m for Earth's atmosphere)
        """
        P0 = 101325.0  # Pa
        H = 8434.0     # m
        pressure = P0 * np.exp(-z_up_m / H)
        self.sensor_data.abs_pressure = pressure
        return pressure

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
        w = tuple(state['w'])

        x, y, z, wq = state['q']   # [i, j, k, w]
        q_send = (wq, x, y, z)

        # Convert local ENU (meters) -> geodetic expected by MAVLink (degE7, mm)
        lat_e7, lon_e7, alt_mm = self._local_enu_to_geodetic(state['x'])

        # Convert velocities: ENU m/s -> NED cm/s expected by MAVLink fields (vx,vy,vz)
        # convention: vx = north, vy = east, vz = down (cm/s)
        v_enu = state['v']
        v_n = float(v_enu[1])
        v_e = float(v_enu[0])
        v_d = float(-v_enu[2])  # z_up -> down
        vx_cms = int(np.round(v_n * 100.0))
        vy_cms = int(np.round(v_e * 100.0))
        vz_cms = int(np.round(v_d * 100.0))

        statedot = self.statedot(state, control, t_step)

        a_enu = statedot["vdot"]
        a_ned = np.array([a_enu[1], a_enu[0], -a_enu[2]], dtype=float)
        a_ned_milli_g = np.clip(np.round(a_ned / 9.80665 * 1000.0), INT_MIN, INT_MAX).astype(np.int16)

        omega = state['w']
        omega_ned = np.array([omega[1], omega[0], -omega[2]], dtype=float)

        # Simulate magnetic field in body frame based on orientation
        mag_field_vector = self.simulate_magnetic_field(state['q'])

        self.conn.mav.hil_state_quaternion_send(
            ts,
            q_send,
            *w,                 # roll/pitch/yaw rates (rad/s)
            lat_e7, lon_e7, alt_mm,
            vx_cms, vy_cms, vz_cms,
            0, 0,               # IAS/TAS as uint16 cm/s
            int(a_ned_milli_g[0]), int(a_ned_milli_g[1]), int(a_ned_milli_g[2])
        )

        self.conn.mav.hil_sensor_send(
            ts,
            *tuple(a_ned),
            *tuple(omega_ned),
            *tuple(mag_field_vector),
            self.simulate_pressure(state['x'][2]),
            0,
            sd.pressure_alt,
            25, # Temperature (°C)
            fields_updated=0xFFFFFFFF
        )

        return state

from datetime import time
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.sensors.imu import Imu
from pymavlink import mavutil
import numpy as np
import types
from pymavlink.dialects.v20.ardupilotmega import MAVLink
from rotorpy.vehicles.px4_params.sihsim_quadx import sihsim_quadx

# Constants
R_EARTH = 6378137.0  # meters
rad2deg = 180.0 / np.pi
INT_MAX = 32767
INT_MIN = -32768

def _compute_hover_rotor_speeds(mass, k_eta, num_rotors, g=9.81):
    """Solve N·k_eta·ω² = m·g for ω and return an array length num_rotors."""
    omega = np.sqrt((mass * g) / (num_rotors * k_eta))
    return np.full(num_rotors, omega)

class PX4Multirotor(Multirotor):
    """PX4 Multirotor Vehicle Model

    Args:
        quad_params (dict): Quadrotor parameters.
        initial_state (dict, optional): Initial state of the quadrotor.
        autopilot_controller (bool): Whether to use the autopilot controller or not.
    """
    def __init__(
        self,
        quad_params=sihsim_quadx,
        initial_state=None,
        control_abstraction="cmd_motor_speeds",
        aero=True,
        enable_ground=True,
        mavlink_url="tcpin:localhost:4560",
        autopilot_controller=True,
        lockstep=True
    ):
        # If no initial state passed, initialize to hover at origin
        if initial_state is None:
            initial_state = {
                'x': np.zeros(3),
                'v': np.zeros(3),
                'q': np.array([0, 0, 0, 1]),
                'w': np.zeros(3),
                'wind': np.zeros(3),
                'rotor_speeds': _compute_hover_rotor_speeds(
                    quad_params['mass'], quad_params['k_eta'], quad_params['num_rotors']
                ),
            }
            initial_state['rotor_speeds'] = np.zeros(quad_params['num_rotors'])
        super().__init__(
            quad_params=quad_params,
            initial_state=initial_state,
            control_abstraction=control_abstraction,
            aero=aero,
            enable_ground=enable_ground,
        )
        # Simulated IMU (with noise)
        self.imu = Imu()
        self._enable_imu_noise = True  # Always add a bit of noise to avoid stale detection
        self.sensor_data = types.SimpleNamespace(
            accel=np.zeros(3),
            gyro=np.zeros(3),
            mag=np.zeros(3),
            abs_pressure=0.0,
            diff_pressure=0.0,
            pressure_alt=0.0,
            temperature=0.0,
        )
        self.t = 0.0
        print("[DEBUG]: PX4Multirotor: Initializing MAVLink connection... on {}".format(mavlink_url))
        self.conn = mavutil.mavlink_connection(mavlink_url)
        self.conn.wait_heartbeat()
        print("[DEBUG]: PX4Multirotor: MAVLink connection established.")

        self._autopilot_controller = autopilot_controller
        self._lockstep = lockstep

        # Try to capture an initial geodetic reference from PX4 (lat/lon in degE7, alt in mm)
        self.lat0_e7 = None
        self.lon0_e7 = None
        self.alt0_mm = None

        # Prefer HOME_POSITION or GLOBAL_POSITION_INT if available quickly
        self._init_geodetic_reference()

    def _init_geodetic_reference(self):
        """
        Poll PX4 for initial geodetic reference (lat/lon in degE7, alt in mm).
        Sets self.lat0_e7, self.lon0_e7, self.alt0_mm.
        """
        self.conn.mav.request_data_stream_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,  # 10 Hz request (best-effort)
            1
        )
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

    def simulate_magnetic_field(self, q, noise_std=0.1):
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

    def simulate_pressure(self, z_up_m, noise_std_pascal=1.5):
        """
        Simulate barometric pressure (Pa) at altitude z_up_m (meters above sea level, z=0 at sea level, z up positive).
        Adds Gaussian noise to simulate sensor noise.
        Uses a simple exponential model: P = P0 * exp(-z/H)
        P0: sea level standard pressure (101325 Pa)
        H: scale height (~8434 m for Earth's atmosphere)
        """
        P0 = 101325.0  # Pa
        H = 8434.0     # m
        pressure = P0 * np.exp(-z_up_m / H)
        # Add Gaussian noise (Pa)
        noisy_pressure = pressure + np.random.normal(0, noise_std_pascal)
        self.sensor_data.abs_pressure = noisy_pressure
        return noisy_pressure

    def _fetch_latest_px4_control(self):
        """Fetch the latest HIL_ACTUATOR_CONTROLS message from PX4 and update control inputs."""

        msg = self.conn.recv_match(type='HIL_ACTUATOR_CONTROLS')
        if msg is not None:
            return {'cmd_motor_speeds': list(msg.controls[:self.num_rotors])}

    def _enu_to_ned_cmps(self, v_enu):
        v_n = float(v_enu[1])
        v_e = float(v_enu[0])
        v_d = float(-v_enu[2])
        return (
            int(np.round(v_n * 100.0)),
            int(np.round(v_e * 100.0)),
            int(np.round(v_d * 100.0)),
        )

    def _imu_and_mag(self, state, statedot):
        meas = self.imu.measurement(state, statedot, with_noise=self._enable_imu_noise)
        a_enu = meas["accel"]
        omega_enu = meas["gyro"]
        # ENU -> NED
        a_ned = np.array([a_enu[1], a_enu[0], -a_enu[2]], dtype=float)
        omega_ned = np.array([omega_enu[1], omega_enu[0], -omega_enu[2]], dtype=float)
        # Keep for logging
        self.sensor_data.accel = a_enu
        self.sensor_data.gyro = omega_enu
        # Magnetometer in body frame (gauss)
        mag_body = self.simulate_magnetic_field(state['q'])
        return a_ned, omega_ned, mag_body

    def _send_hil_packets(self, ts, state, control):
        x, y, z, wq = state['q']
        q_send = (wq, x, y, z)
        lat_e7, lon_e7, alt_mm = self._local_enu_to_geodetic(state['x'])
        vx_cms, vy_cms, vz_cms = self._enu_to_ned_cmps(state['v'])

        statedot = self.statedot(state, control, 0.0)
        a_ned, omega_ned, mag_body = self._imu_and_mag(state, statedot)

        a_ned_mg = np.clip(np.round(a_ned / 9.80665 * 1000.0), INT_MIN, INT_MAX).astype(np.int16)

        self.conn.mav.hil_state_quaternion_send(
            ts,
            q_send,
            *tuple(state['w']),
            lat_e7, lon_e7, alt_mm,
            vx_cms, vy_cms, vz_cms,
            0, 0,
            int(a_ned_mg[0]), int(a_ned_mg[1]), int(a_ned_mg[2])
        )

        # Only flag accel/gyro as updated (exclude mag and baro-related fields)

        flags_accel = 1 | 2 | 4           # XACC | YACC | ZACC
        flags_gyro  = 8 | 16 | 32         # XGYRO | YGYRO | ZGYRO
        updated_mask = flags_accel | flags_gyro  # = 63

        self.conn.mav.hil_sensor_send(
            ts,
            *tuple(a_ned),
            *tuple(omega_ned),
            *tuple(mag_body),
            self.simulate_pressure(state['x'][2]),
            0,
            self.sensor_data.pressure_alt,
            25,
            fields_updated=updated_mask,
        )

    def step(self, state, control, t_step):
        ts = int(self.t * 1e6)
        self._send_hil_packets(ts, state, control)
        
        # Use PX4 commands only if autopilot_controller is True
        if self._autopilot_controller:
            px4_control = self._fetch_latest_px4_control(blocking=self._lockstep_enabled)
            if px4_control is not None:
                control = px4_control

        state = super().step(state, control, t_step)
        self.state = state
        self.t += t_step

        return state

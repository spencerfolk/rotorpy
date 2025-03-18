"""This class inherits from Multirotor and is used to interface with Ardupilot SITL.
It sends the state to the SITL to verify that the conversion is correct.
"""
import copy
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.sensors.imu import Imu
from rotorpy.vehicles.hummingbird_params import quad_params as crazyflie_params
from scipy.spatial.transform import Rotation as R

try:
    from ArduPilotPlugin import ArduPilotPlugin
except:
    print("Missing dependency ArduPilotPlugin. Install it from source at https://github.com/TomerTip/PyArduPilotPlugin")

PWM_MIN = 1100
PWM_MAX = 1900
SEND_RATE_HZ = 1000

@dataclass
class ControlCommand:
    cmd_motor_speeds: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
        
    def to_dict(self) -> dict:
        return asdict(self)

x0 = {'x': np.array([0,0,0]),
      'v': np.zeros(3,),
      'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
      'w': np.zeros(3,),
      'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
      'rotor_speeds': np.array([0.0, 0.0, 0.0, 0.0])}


@dataclass
class SensorData:
    sim_position : List[float]
    sim_attitude : List[float]
    sim_velocity_inertial : List[float]
    xgyro: float
    ygyro: float
    zgyro: float
    xacc: float
    yacc: float 
    zacc: float
    
    def __post_init__(self):
        assert len(self.sim_position) == 3
        assert len(self.sim_velocity_inertial) == 3
        assert len(self.sim_attitude) == 4


class Ardupilot(Multirotor):
    M_glu2frd = R.from_euler('x', np.pi)
    M_enu2ned = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    def __init__(self, quad_params, initial_state = x0, control_abstraction='cmd_motor_speeds', aero=True, ardupilot_control=False, enable_imu_noise = False, enable_ground = False):
        super().__init__(quad_params, initial_state, control_abstraction, aero, enable_ground)
        self._enable_ground = enable_ground
        self._ardupilot_control = ardupilot_control
        statedot = {'vdot' : np.zeros(3,), 'wdot' : np.zeros(3,)}
        self.t = 0.0
        self.imu = Imu()
        self.__enable_imu_noise = enable_imu_noise

        self.sensor_data = self._create_sensor_data(self.initial_state, statedot, self.imu, self.__enable_imu_noise)
        self.ap_link = ArduPilotPlugin()
        self.ap_link.drain_unread_packets()

        self._control_cmd = ControlCommand()   


    def step(self, state, control, t_step):
        received, pwm = self.ap_link.pre_update(self.t)
        if self._ardupilot_control:
            control = {'cmd_motor_speeds': self._motor_cmd_to_omega(self._control_cmd.cmd_motor_speeds)}

        # TODO: this should be moved inside the `Multirotor` class
        if self._on_ground(state) and self._enable_ground:
            state = self._handle_vehicle_on_ground(state)

        statedot = self.statedot(state, control, t_step)
        state =  super().step(state, control, t_step)
        self.t += t_step

        self.sensor_data = self._create_sensor_data(state, statedot, self.imu, self.__enable_imu_noise)
        
        self.ap_link.post_update(self.sensor_data, self.t)
        if received: # TODO: is this being handled correctly?
            self._control_cmd.cmd_motor_speeds = list(pwm[0:4]) # type: ignore
        
        return state

    def _handle_vehicle_on_ground(
        self, state: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Handles the vehicle's state when it is on the ground.
        This method performs the following actions:
        - Constrains the vehicle's position to the ground level (z = 0).
        - Stops any downward vertical motion by setting the vertical velocity to zero if it is negative.
        - Resets the angular velocity to zero to stop any spinning motion.
        - Sets the pitch and roll angles to zero while preserving the heading angle.
        Args:
            state (Dict[str, np.ndarray]): The current state of the vehicle, which includes position ('x'),
                                           velocity ('v'), angular velocity ('w'), orientation ('q'),
                                           wind vector ('wind') and motor angular velocities ('rotor_speeds').
        Returns:
            Dict[str, np.ndarray]: The updated state of the vehicle after applying the ground constraints.
        """
        # FIXME: when the motor command is zero and the vehicle is on the ground it drifts (threshold issue?)
        state["x"][2] = 0

        if state["v"][2] < 0:
            state["v"] = np.zeros(3,)

        state["w"] = np.zeros(
            3,
        )

        state["q"] = flatten_attitude(state["q"])

        return state

    @staticmethod
    def _motor_cmd_to_omega(pwm_commands : List[int]) -> List[float]:
        """Convert the pwm commands received from the SITL into angular velocity targets for the motors.

        Args:
            pwm_commands (List[int]): pwm commands [1000-2000] output by Ardupilot SITL

        Returns:
            List[float]: angular velocities targets for the motors.
        """
        rotor_0, rotor_1, rotor_2, rotor_3 = pwm_commands
        reordered_pwm_commands = [rotor_2, rotor_0, rotor_3, rotor_1]
        normalized_commands = [(c-PWM_MIN)/(PWM_MAX-PWM_MIN) for c in reordered_pwm_commands]
        angular_velocities = [838.0*c for c in normalized_commands] # TODO: remove magic constant
        return angular_velocities      

    @staticmethod
    def _create_sensor_data(
        state: Dict[str, np.ndarray],
        statedot: Dict[str, np.ndarray],
        imu: Imu,
        enable_imu_noise: bool = False,
    ) -> SensorData:
        """
        Converts the state and state derivative of a `rotorpy.vehicles.multirotor` object to the Ardupilot convention
        for position, velocity, acceleration, quaternions, and angular velocities.
        Args:
            state (Dict[str, np.ndarray]): The current state of the multirotor, including position, velocity, and attitude quaternion.
            statedot (np.ndarray): The state derivative, representing the acceleration.
            imu (Imu): The IMU object used to obtain measurements.
            enable_imu_noise (bool, optional): Flag to enable or disable IMU noise in the measurements. Defaults to False.
        Notes:
            - The attitude quaternion in rotorpy is in scalar-last representation (x, y, z, w) and represents the rotation
              from the body frame (GLU) to the world frame (ENU).
            - The attitude quaternion in Ardupilot is in scalar-first representation (w, x, y, z) and represents the rotation
              from the world frame (NED) to the body frame (FRD).

        Returns:
            SensorData: The sensor data in the Ardupilot convention, including position, velocity, acceleration,
                        attitude quaternion, and angular velocities.
        """

        # 1. Obtain attitude quaternion (scalar-first),
        # representing the rotation from the body (GLU) frame to the world (ENU) frame
        R_glu2enu = R.from_quat(state["q"], scalar_first=False)

        # 2. Obtain the IMU meaurements in the GLU frame
        acceleration = copy.deepcopy(statedot)
        meas_dict = imu.measurement(state, acceleration, with_noise=enable_imu_noise)
        a_glu, omega_glu = meas_dict["accel"], meas_dict["gyro"]
        a_frd = Ardupilot.M_glu2frd.apply(a_glu).tolist()
        omega_frd = Ardupilot.M_glu2frd.apply(omega_glu).tolist()

        # 2. Obtain the rotation from the body frame (FRD) to the world frame (NED)
        # This is the attitude of the FRD frame in the NED frame
        R_frd2ned = Ardupilot.M_enu2ned * R_glu2enu * Ardupilot.M_glu2frd

        return SensorData(
            state["x"].tolist(),
            R_frd2ned.as_quat(scalar_first=True).tolist(),
            state["v"].tolist(),
            xgyro=omega_frd[0],
            ygyro=omega_frd[1],
            zgyro=omega_frd[2],
            xacc=a_frd[0],
            yacc=a_frd[1],
            zacc=a_frd[2],
        )


def flatten_attitude(quaternion : List[float]) -> List[float]:
    """
    Set roll and pitch to 0 while keeping yaw unchanged.
    
    Parameters:
        quaternion (array-like): Quaternion [x, y, z, w] representing the quadrotor's attitude.
    
    Returns:
        numpy.ndarray: New quaternion with roll and pitch set to 0.
    """

    # Extract Euler angles in the 'XYZ' (roll, pitch, heading) convention wrt the world frame
    _, _, heading = R.from_quat(quaternion).as_euler('XYZ', degrees=False)
    
    # Create a new rotation object with roll and pitch set to 0
    flattened_rotation = R.from_euler('Z', heading, degrees=False)
    
    # Convert the new rotation back to a quaternion
    return flattened_rotation.as_quat()

if __name__ == '__main__':
    r = R.from_euler('y', 0, degrees=True)
    initial_state = {'x': np.array([0,0,0]),
                                            'v': np.zeros(3,),
                                            'q': r.as_quat(scalar_first=False), # [i,j,k,w]
                                            'w': np.zeros(3,),
                                            'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                            'rotor_speeds': np.array([0, 0, 0, 0])}
    vehicle = Ardupilot(initial_state=initial_state, quad_params = crazyflie_params, ardupilot_control=True, enable_imu_noise=False, enable_ground=True)

    state = initial_state

    dt = 0.01
    while True:
        state = vehicle.step(state, {'cmd_motor_speeds': [0,]*4}, dt)
        time.sleep(dt)

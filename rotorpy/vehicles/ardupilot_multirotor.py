"""This class inherits from Multirotor and is used to interface with Ardupilot SITL.
It sends the state to the SITL to verify that the conversion is correct.
"""
import copy
from dataclasses import asdict, dataclass, field
import threading
import time
from typing import Dict, List

import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.sensors.imu import Imu
from rotorpy.vehicles.hummingbird_params import quad_params as crazyflie_params
from scipy.spatial.transform import Rotation as R

from ArduPilotPlugin import ArduPilotPlugin

PWM_MIN = 1100
PWM_MAX = 1900
SEND_RATE_HZ = 600

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


def enu2ned(x_enu,y_enu,z_enu):
    return [y_enu, x_enu, -z_enu]

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
        self.t = 0
        self.imu = Imu()
        self.__enable_imu_noise = enable_imu_noise

        self.sensor_data = self._create_sensor_data(self.initial_state, statedot, self.imu, self.__enable_imu_noise)
        self.ap_link = ArduPilotPlugin()
        self.ap_link.drain_unread_packets()

        self._control_cmd = ControlCommand()   

        self._sitl_output_thread = threading.Thread(target=self._send_loop, name='SITL_out')
        self._sitl_output_thread.start()

    def step(self, state, control, t_step):
        if self._ardupilot_control:
            control = {'cmd_motor_speeds': self._motor_cmd_to_omega(self._control_cmd.cmd_motor_speeds)}

        # TODO: this should be moved inside the `Multirotor` class
        if self._on_ground(state) and self._enable_ground:
            state = self._handle_vehicle_on_ground(state)

        statedot = self.statedot(state, control, t_step)
        state =  super().step(state, control, t_step)
        self.t += t_step

        self.sensor_data = self._create_sensor_data(state, statedot, self.imu, self.__enable_imu_noise)

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

        state["x"][2] = 0

        if state["v"][2] < 0:
            state["v"][2] = 0

        state["w"] = np.zeros(
            3,
        )

        _, _, heading = R.from_quat(state["q"]).as_euler("XYZ")
        state["q"] = R.from_euler("Z", heading).as_quat()

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
    def _create_sensor_data(state : Dict[str, np.ndarray], statedot, imu : Imu, enable_imu_noise = False):
        """ TODO: improve docstring
        This function takes the state and state derivative of a `rotorpy.vehicles.multirotor` object and converts it to the Ardupilot convention 
        for position, velocity, acceleration, quaternions and angular velocities.
        
        The attitude quaternion in rotorpy is in scalar-last representation (x,y,z,w)
        It represents the rotation from the body frame (GLU) to the world frame (ENU)
        The attitude quaternion in Ardupilot is in scalar-first representation (w,x,y,z)
        It represents the rotation from the world frame (NED) to the body frame (FRD) 
        
        Returns:
            SensorData
        """

        # 1. Obtain attitude quaternion (scalar-first),
        # representing the rotation from the body (GLU) frame to the world (ENU) frame
        R_glu2enu = R.from_quat(state['q'], scalar_first = False)

        # 2. Obtain the IMU meaurements in the GLU frame
        acceleration = copy.deepcopy(statedot)
        meas_dict = imu.measurement(state, acceleration, with_noise=enable_imu_noise)
        a_glu, omega_glu = meas_dict['accel'], meas_dict['gyro']
        a_frd = Ardupilot.M_glu2frd.apply(a_glu).tolist()
        omega_frd = Ardupilot.M_glu2frd.apply(omega_glu).tolist()

        # 2. Obtain the rotation from the body frame (FRD) to the world frame (NED)
        # This is the attitude of the FRD frame in the NED frame
        R_frd2ned = Ardupilot.M_enu2ned * R_glu2enu * Ardupilot.M_glu2frd      

        return SensorData(
            enu2ned(*state['x'].tolist()),\
            R_frd2ned.as_quat(scalar_first=True).tolist(),\
            enu2ned(*state['v'].tolist()),\
            xgyro=omega_frd[0],
            ygyro=omega_frd[1],
            zgyro=omega_frd[2],
            xacc=a_frd[0],
            yacc=a_frd[1],
            zacc=a_frd[2]
        )

    def _send_loop(self):
        """
        Ardupilot needs to receive packets from the simulator at a constant rate:
        As per https://github.com/ArduPilot/ardupilot/blob/f236a6c6fcac112a2271763a344634302d65da82/libraries/SITL/examples/JSON/readme.md:
        
        ```
        The frame rate represents the time step the simulation should take, 
        this can be changed with the SIM_RATE_HZ ArduPilot parameter. 
        The physics backend is free to ignore this value, 
        a maximum time step size would typically be set. 
        The SIM_RATE_HZ should value be kept above the vehicle loop rate, 
        by default this 400hz on copter and quadplanes and 50 hz on plane and rover.
        ```
        """
        interval = 1.0 / SEND_RATE_HZ
        while True:
            start_time = time.time()
            received, pwm = self.ap_link.pre_update(self.t)
            if received: # TODO: is this being handled correctly?
                self._control_cmd.cmd_motor_speeds = list(pwm[0:4])
            self.ap_link.post_update(self.sensor_data, self.t)
            elapsed_time = time.time() - start_time
            time.sleep(max(0, interval - elapsed_time))

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
        print(f"\nAttitude angles: {R.from_quat(state['q'], scalar_first=True).as_euler('zyx', degrees=True)}\n Height: {state['x'][2]}\n")
        time.sleep(dt)

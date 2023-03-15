import numpy as np
from scipy.spatial.transform import Rotation
import copy

class Imu:
    """
    Simulated IMU measurement given
      1) quadrotor's ground truth state and acceleration, and
      2) IMU's pose in quadrotor body frame.

      CREDIT:
      Partial implementation from Yu-Ming Chen and MEAM 620 Sp2022 Teaching Staff
      Finishing touches added by Alexander Spinos, checked by Spencer Folk.
    """
    def __init__(self, accelerometer_params={'initial_bias': np.array([0.,0.,0.]), # m/s^2
                                             'noise_density': (0.38**2)*np.ones(3,), # m/s^2 / sqrt(Hz)
                                             'random_walk': np.zeros(3,) # m/s^2 * sqrt(Hz)
                                             }, 
                gyroscope_params={'initial_bias': np.array([0.,0.,0.]), # m/s^2
                                  'noise_density': (0.01**2)*np.ones(3,), # rad/s / sqrt(Hz)
                                  'random_walk': np.zeros(3,) # rad/s * sqrt(Hz)
                                  }, 
                R_BS = np.eye(3),
                p_BS = np.zeros(3,),
                sampling_rate=500, 
                gravity_vector=np.array([0,0,-9.81])):
        """
        Parameters:
            R_BS, the rotation matrix of sensor frame S in body frame B
            p_BS, the position vector from frame B's origin to frame S's origin, expressed in frame B
            accelerometer_params, a dict with keys:
                initial_bias, accelerometer contant bias,    [ m / s^2 ]
                noise_density, accelerometer "white noise",  [ m / s^2 / sqrt(Hz) ]
                random_walk, accelerometer bias diffusion,   [ m / s^2 * sqrt(Hz) ]
            gyroscope_params, a dict with keys:
                initial_bias, gyro contant bias,             [ m / s^2 ]
                noise_density, gyro "white noise",           [ rad / s / sqrt(Hz) ]
                random_walk, gyro bias diffusion,            [ rad / s * sqrt(Hz) ]
            sampling_rate, the sampling rate of the sensor, Hz (1/s)
            gravity_vector, the gravitational vector in world frame (should be ~ [0, 0 , -9.81])
        """

        # A few checks
        if type(R_BS) != np.ndarray:
            raise TypeError("R_BS's type is not numpy.ndarray")
        if type(p_BS) != np.ndarray:
            raise TypeError("p_BS's type is not numpy.ndarray")
        if type(gravity_vector) != np.ndarray:
            raise TypeError("gravity_vector's type is not numpy.ndarray")
        if R_BS.shape != (3, 3):
            raise ValueError("R_BS's size is not (3, 3)")
        if p_BS.shape != (3,):
            raise ValueError("p_BS's size is not (3,)")
        if gravity_vector.shape != (3,):
            raise ValueError("gravity_vector's size is not (3,)")

        self.R_BS = R_BS
        self.p_BS = p_BS
        self.rate_scale = np.sqrt(sampling_rate/2)
        self.gravity_vector = gravity_vector

        self.accel_variance = accelerometer_params['noise_density'].astype('float64')
        self.accel_random_walk = accelerometer_params['random_walk'].astype('float64')
        self.accel_bias = accelerometer_params['initial_bias'].astype('float64')
        self.gyro_variance = gyroscope_params['noise_density'].astype('float64')
        self.gyro_random_walk = gyroscope_params['random_walk'].astype('float64')
        self.gyro_bias = gyroscope_params['initial_bias'].astype('float64')

    def bias_step(self):
        """Simulate bias drift"""

        self.accel_bias += np.random.normal(scale=self.accel_random_walk) / self.rate_scale
        self.gyro_bias += np.random.normal(scale=self.gyro_random_walk) / self.rate_scale

        return

    def measurement(self, state, acceleration, with_noise=True):
        """
        Computes and returns the IMU measurement at a time step.

        Inputs:
            state, a dict describing the state with keys
                x, position, m, shape=(3,)
                v, linear velocity, m/s, shape=(3,)
                q, quaternion [i,j,k,w], shape=(4,)
                w, angular velocity (in LOCAL frame!), rad/s, shape=(3,)
            acceleration, a dict describing the acceleration with keys
                vdot, quadrotor's linear acceleration expressed in world frame, m/s^2, shape=(3,)
                wdot, quadrotor's angular acceleration expressed in world frame, rad/s^2, shape=(3,)
        Outputs:
            observation, a dict describing the IMU measurement with keys
                accel, simulated accelerometer measurement, m/s^2, shape=(3,)
                gyro, simulated gyroscope measurement, rad/s^2, shape=(3,)
        """
        q_WB = state['q']
        w_WB = state['w']
        alpha_WB_W = acceleration['wdot']
        a_WB_W = acceleration['vdot']

        # Rotation matrix of the body frame B in world frame W
        R_WB = Rotation.from_quat(q_WB).as_matrix()

        # Sensor position in body frame expressed in world coordinates
        p_BS_W = R_WB @ self.p_BS

        # Linear acceleration of point S (the imu) expressed in world coordinates W.
        a_WS_W = a_WB_W + np.cross(alpha_WB_W, p_BS_W) + np.cross(w_WB, np.cross(w_WB, p_BS_W))

        # Rotation from world to imu: R_SW = R_SB * R_BW
        R_SW = self.R_BS.T @ R_WB.T

        # Rotate to local frame
        accelerometer_measurement = R_SW @ (a_WS_W - self.gravity_vector)
        gyroscope_measurement = copy.deepcopy(w_WB).astype(float)

        # Add the bias drift (default 0)
        self.bias_step()

        # Add biases and noises
        accelerometer_measurement += self.accel_bias
        gyroscope_measurement += self.gyro_bias
        if with_noise:
            accelerometer_measurement += self.rate_scale * np.random.normal(scale=np.abs(self.accel_variance))
            gyroscope_measurement +=  self.rate_scale * np.random.normal(scale=np.abs(self.gyro_variance))

        return {'accel': accelerometer_measurement, 'gyro': gyroscope_measurement}

if __name__=="__main__":
    # The default 
    sim_rate = 100
    accelerometer_params = {'initial_bias': np.array([0,0,0]), # m/s^2
                                    'noise_density': (0.38**2)*np.ones(3,), # m/s^2 / sqrt(Hz)
                                    'random_walk': np.zeros(3,) # m/s^2 * sqrt(Hz)
                                    }
    gyroscope_params     = {'initial_bias': np.array([0,0,0]), # m/s^2
                            'noise_density': (0.01**2)*np.ones(3,), # rad/s / sqrt(Hz)
                            'random_walk': np.zeros(3,) # rad/s * sqrt(Hz)
                            }

    imu = Imu(accelerometer_params,
                gyroscope_params,
                p_BS = np.zeros(3,),
                R_BS = np.eye(3),
                sampling_rate=sim_rate)

    print(imu.measurement({'x': np.array([0,0,0]), 'v': np.array([0,0,0]), 'q': np.array([0,0,0,1]), 'w': np.array([0,0,0])}, 
                    {'vdot': np.array([0,0,0]), 'wdot': np.array([0,0,0])}))
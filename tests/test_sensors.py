'''
Test the basic sensors to ensure they are functioning as expected.
'''

import numpy as np

IMU_OUTPUT_SIGNATURE = {
    'accel': np.ndarray,  # Acceleration vector (m/s^2)
    'gyro': np.ndarray,  # Angular velocity vector (rad/s)
}

MOCAP_OUTPUT_SIGNATURE = {
    'x': np.ndarray,  # Position vector (m)
    'v': np.ndarray,  # Velocity vector (m/s)
    'q': np.ndarray,  # Quaternion (qx, qy, qz, qw)
    'w': np.ndarray,  # Angular velocity vector (rad/s)
}

def test_imu():
    from rotorpy.sensors.imu import Imu

    print("\nTesting IMU class")

    # Create an instance of the IMU class
    imu_obj = Imu()

    # Check if the class has some important attributes that are used elsewhere
    print("\tChecking for important attributes...")
    params = ['accel_bias', 'gyro_bias', 'accel_variance', 'gyro_variance', 'rate_scale', 'gravity_vector', 'p_BS', 'R_BS']
    for param in params:
        assert hasattr(imu_obj, param), f"IMU class is missing '{param}' attribute"

    # Take one measurement and check if the output is as expected  
    # TODO: This test does not check the accuracy of the IMU model... change p_BS and R_BS and assert that the measurement matches the theoretical value. 
    print("\tChecking if measurement() method works as intended...")
    state = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0, 0, 0, 1]), 'w': np.zeros(3)}
    acceleration = {'vdot': np.zeros(3), 'wdot': np.zeros(3)}
    measurement = imu_obj.measurement(state, acceleration, with_noise=False)
    assert isinstance(measurement, dict), "measurement() method should return a dictionary"
    for key in IMU_OUTPUT_SIGNATURE.keys():
        assert key in measurement, f"Measurement should have '{key}' key"
        assert isinstance(measurement[key], IMU_OUTPUT_SIGNATURE[key]), f"Measurement[{key}] should be a {IMU_OUTPUT_SIGNATURE[key]}"

def test_external_mocap():
    from rotorpy.sensors.external_mocap import MotionCapture

    print("\nTesting MotionCapture class")

    # Create an instance of the MotionCapture class
    mocap_params={'pos_noise_density':  np.zeros(3),  # noise density for position 
                  'vel_noise_density':  np.zeros(3),          # noise density for velocity
                  'att_noise_density':  np.zeros(3),          # noise density for attitude 
                  'rate_noise_density': np.zeros(3),         # noise density for body rates
                  'vel_artifact_max': 5,                              # maximum magnitude of the artifact in velocity (m/s)
                  'vel_artifact_prob': 0,                         # probability that an artifact will occur for a given velocity measurement
                  'rate_artifact_max': 1,                             # maximum magnitude of the artifact in body rates (rad/s)
                  'rate_artifact_prob': 0                        # probability that an artifact will occur for a given rate measurement
                }
    mocap_obj = MotionCapture(sampling_rate=100, mocap_params=mocap_params, with_artifacts=False)

    # Check if the class has some important attributes that are used elsewhere
    print("\tChecking for important attributes...")
    params = ['rate_scale', 'pos_density', 'vel_density', 'att_density', 'rate_density', 'vel_artifact_max', 'vel_artifact_prob', 'rate_artifact_max', 'rate_artifact_prob', 'initialized', 'with_artifacts']
    for param in params:
        assert hasattr(mocap_obj, param), f"MotionCapture class is missing '{param}' attribute"
    
    # Take one measurement and check if the output is as expected
    print("\tChecking if measurement() method works as intended...")
    state_dict = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0, 0, 0, 1]), 'w': np.zeros(3)}
    measurement = mocap_obj.measurement(state_dict, with_noise=False, with_artifacts=False)
    assert isinstance(measurement, dict), "measurement() method should return a dictionary"
    for key in MOCAP_OUTPUT_SIGNATURE.keys():
        assert key in measurement, f"Measurement should have '{key}' key"
        assert isinstance(measurement[key], MOCAP_OUTPUT_SIGNATURE[key]), f"Measurement[{key}] should be a {MOCAP_OUTPUT_SIGNATURE[key]}"

    for state in state_dict.keys():
        assert np.allclose(state_dict[state], measurement[state]), f"Measurement for {state} should be the same as the state"

def test_range_sensors():
    from rotorpy.sensors.range_sensors import TwoDRangeSensor
    from rotorpy.world import World

    print("\nTesting TwoDRangeSensor class")

    world = World.empty(extents=[-10, 10, -10, 10, -10, 10])

    # Create an instance of the TwoDRangeSensor class
    range_sensor = TwoDRangeSensor(world, sampling_rate=100, map_resolution=1, angular_resolution=1, angular_fov=360, Dmin=0.025, Dmax=100, noise_density=0.0, fixed_heading=True)

    # Check if the class has some important attributes that are used elsewhere
    print("\tChecking for important attributes...")
    params = ['world', 'occ_map', 'rate_scale', 'angular_resolution', 'angular_fov', 'Dmin', 'Dmax', 'sensor_variance', 'fixed_heading', 'ray_angles', 'N_rays', 'world_edges']
    for param in params:
        assert hasattr(range_sensor, param), f"TwoDRangeSensor class is missing '{param}' attribute"
    
    # Take one measurement and check if the output is as expected
    # TODO: This test does not check the accuracy of the range sensor model... change the world and assert that the measurement matches the theoretical value.
    print("\tChecking if measurement() method works as intended...")
    state_dict = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0, 0, 0, 1]), 'w': np.zeros(3)}
    measurement = range_sensor.measurement(state_dict, with_noise=False)
    assert isinstance(measurement, np.ndarray), "measurement() method should return a dictionary"
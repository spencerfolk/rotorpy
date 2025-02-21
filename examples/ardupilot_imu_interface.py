from rotorpy.sensors.imu import Imu
import numpy as np

if __name__ == '__main__':
    imu = Imu()

    """
    If the drone is flat on a surface the accelerometer of the imu senses the counteracting force of the surface pushing up against gravity.
    In the GLU body frame convention, down is positive, so the upward force due to gravity appears as a vector [0, 0, 9.81] m/s^2.
    """

    state = {'x': np.array([0,0,0]), 'v': np.array([0,0,0]), 'q': np.array([0,0,0,1]), 'w': np.array([0,0,0])}
    acceleration = {'vdot': np.array([0,0,0]), 'wdot': np.array([0,0,0])}
    print(imu.measurement(state, acceleration, with_noise=False))

    from scipy.spatial.transform import Rotation as R
    # Ardupilot uses the FRD reference frame. The transformation from GLU to FRD is a rotation of 180deg around the x body axis:
    T_glu2frd = R.from_euler('x', 180, degrees=True)
    imu_frd = Imu(R_BS=T_glu2frd.as_matrix())

    print("Rotate the drone of 90 degrees about the world X axis (ENU)")
    rotation_x = R.from_euler('x', 90, degrees=True)
    state['q'] = rotation_x.as_quat(scalar_first=False)
    imu_measurement=imu.measurement(state, acceleration, with_noise=False)
    print(imu_measurement)
    assert np.allclose(np.array([0, 9.81, 0]), imu_measurement['accel'])
    # Verify the vector in FRD body frame
    assert np.allclose(np.array([0, -9.81, 0]), T_glu2frd.as_matrix()@imu_measurement['accel'])
    imu_frd_measurement=imu_frd.measurement(state, acceleration, with_noise=False)
    assert np.allclose(np.array([0, -9.81, 0]), imu_frd_measurement['accel'])
    # assert np.allclose(rot_glu2frd.as_matrix()@imu_measurement['accel'], rot_glu2frd.apply(imu_measurement['accel']))

    print("Rotate the drone of 90 degrees about the world Y axis (ENU)")
    rotation_y = R.from_euler('y', 90, degrees=True)
    state['q'] = rotation_y.as_quat(scalar_first=False)
    imu_measurement=imu.measurement(state, acceleration, with_noise=False)
    print(imu_measurement)
    assert np.allclose(np.array([-9.81, 0, 0]), imu_measurement['accel'])
    assert np.allclose(np.array([-9.81, 0, 0]), T_glu2frd.as_matrix()@imu_measurement['accel'])
    imu_frd_measurement=imu_frd.measurement(state, acceleration, with_noise=False)
    assert np.allclose(np.array([-9.81, 0, 0]), imu_frd_measurement['accel'])
    
    print("Rotate the drone of 90 degrees about the world Z axis (ENU)")
    rotation_z = R.from_euler('z', 90, degrees=True)
    state['q'] = rotation_z.as_quat(scalar_first=False)
    imu_measurement=imu.measurement(state, acceleration, with_noise=False)
    print(imu_measurement)
    assert np.allclose(np.array([0, 0, 9.81]), imu_measurement['accel'])
    imu_frd_measurement=imu_frd.measurement(state, acceleration, with_noise=False)
    assert np.allclose(np.array([0, 0, -9.81]), imu_frd_measurement['accel'])
    
# Sensor Module

Sensors are implemented as ways to convert the current ground truth vehicle state (and possibly obstacles and/or control commands) to measurements that reflects information that might be available in the real world. 

The most straightforward examples of sensors are in `imu.py` and `external_mocap.py` which mimic an inertial measurement unit and external motion capture system (e.g. Vicon), respectively. 

The IMU sensor can be arbitrarily placed and oriented with respect to the body axes. Bias and noise intensity can be specified for each sensor axis, in addition to bias drift (diffusion). 

The external motion capture sensor provides noisy measurements of pose and twist. The sensor is placed at the center of mass aligned with the body axes. There is also a parameter that enables "artifacts", which are the consequence of numerical differentiation or glitches in the system causing spikes in the measurements. 

The camera sensor (`camera.py`) provides a simulated pinhole camera that renders a synthetic image of the world without photorealistic rendering. It relies on visual features placed on world blocks (see `rotorpy/world.py`), each with a 3D position and an RGB descriptor. The camera model supports a fully customizable calibration via `intrinsics` (focal lengths, image width/height, principal point, and `[k1, k2, p1, p2, k3]` radial/tangential distortion coefficients) and `extrinsics` (position and orientation quaternion `[i,j,k,w]` of the camera in the vehicle body frame). Occlusion is handled explicitly: a feature is only rendered if no block lies between it and the camera, so overlapping-object physics are respected. The batched variant `BatchedPinholeCamera` renders from many vehicles simultaneously using torch (CPU or GPU). See `examples/camera_visualization.py` and `tests/test_camera.py` for usage.

See each sensor file for more information. 
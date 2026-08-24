'''
Tests for the integration of the camera sensor into the simulation pipeline
(simulate() and Environment), including frame rate decimation, agreement with
direct camera.measurement() calls, exclusion of images from the CSV, and
saving/loading of camera data.
'''

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for tests
import matplotlib.pyplot as plt


TINY_INTRINSICS = {'fx': 100., 'fy': 100., 'width': 64, 'height': 48, 'cx': 32., 'cy': 24.,
                   'dist_coeffs': np.zeros(5)}

# World with a single block and regular features; a camera below the block at
# [0.5, 0.5, -1] with identity attitude (looking along +z) sees its bottom face.
def _feature_world():
    from rotorpy.world import World
    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 'color': [1.0, 0.0, 0.0]}],
    }
    return World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.25,
                 descriptor_noise=0.1, seed=3)


def _camera(frame_rate=None):
    from rotorpy.sensors.camera import PinholeCamera
    return PinholeCamera(intrinsics=TINY_INTRINSICS,
                         extrinsics={'position': np.zeros(3),
                                     'orientation': np.array([0.0, 0.0, 0.0, 1.0])},
                         frame_rate=frame_rate)


def _initial_state():
    return {'x': np.array([0.5, 0.5, -1.0]),
            'v': np.zeros(3,),
            'q': np.array([0.0, 0.0, 0.0, 1.0]),
            'w': np.zeros(3,),
            'wind': np.zeros(3,),
            'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}


def test_simulate_camera_frame_decimation():
    """
    Frames must be captured at t=0 and then whenever the camera's frame_rate
    dictates, on the simulation time grid.
    """
    print("\nTesting simulate() camera frame rate decimation")
    from rotorpy.simulate import simulate
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.vehicles.crazyflie_params import quad_params
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.trajectories.hover_traj import HoverTraj
    from rotorpy.wind.default_winds import NoWind
    from rotorpy.sensors.imu import Imu
    from rotorpy.sensors.external_mocap import MotionCapture
    from rotorpy.estimators.nullestimator import NullEstimator

    world = _feature_world()
    sim_rate = 100
    t_final = 0.52
    common = dict(initial_state=_initial_state(),
                  vehicle=Multirotor(quad_params),
                  controller=SE3Control(quad_params),
                  trajectory=HoverTraj(x0=[0.5, 0.5, -1.0]),
                  wind_profile=NoWind(),
                  imu=Imu(sampling_rate=sim_rate),
                  mocap=MotionCapture(sampling_rate=sim_rate),
                  estimator=NullEstimator(),
                  t_final=t_final,
                  t_step=1/sim_rate,
                  safety_margin=0.25,
                  use_mocap=False)

    # frame_rate = 20 Hz -> capture every 5th step: 0, 0.05, ..., 0.50.
    cam = _camera(frame_rate=20)
    (time_hist, _, _, _, _, _, _, _, _, cm) = simulate(world, camera=cam, **common)
    expected_times = np.arange(0, t_final, 0.05)
    assert np.allclose(cm['time'], expected_times, atol=1e-9), \
        "camera frames should be captured every 1/frame_rate seconds starting at t=0"
    # Frame timestamps are a subset of the simulation time grid.
    for t in cm['time']:
        assert np.any(np.isclose(time_hist, t, atol=1e-9)), "frame times must lie on the sim grid"

    # frame_rate = None -> render every simulation step.
    cam_every = _camera(frame_rate=None)
    (time_every, _, _, _, _, _, _, _, _, cm_every) = simulate(world, camera=cam_every, **common)
    assert cm_every['time'].shape[0] == time_every.shape[0], \
        "frame_rate None should capture one frame per simulation step"

    # Stacked outputs have consistent shapes; variable-size outputs are lists.
    K = cm['time'].shape[0]
    N = world.get_surface_features().shape[0]
    assert cm['image'].shape == (K,) + (48, 64, 3) and cm['image'].dtype == np.uint8
    assert cm['visible_mask'].shape == (K, N)
    assert cm['projected'].shape == (K, N, 2)
    assert cm['depth'].shape == (K, N)
    assert len(cm['keypoints']) == len(cm['keypoint_depths']) == len(cm['visible_features']) == K

    # No camera -> None.
    (_, _, _, _, _, _, _, _, _, cm_none) = simulate(world, **common)
    assert cm_none is None


def test_simulate_camera_matches_direct_measurement():
    """
    The frames collected during simulate() must equal what a direct call to
    PinholeCamera.measurement() would produce for the same vehicle states.
    """
    print("\nTesting simulate() camera frames vs direct measurement()")
    from rotorpy.simulate import simulate
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.vehicles.crazyflie_params import quad_params
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.trajectories.hover_traj import HoverTraj
    from rotorpy.wind.default_winds import NoWind
    from rotorpy.sensors.imu import Imu
    from rotorpy.sensors.external_mocap import MotionCapture
    from rotorpy.estimators.nullestimator import NullEstimator

    world = _feature_world()
    cam = _camera(frame_rate=None)
    sim_rate = 100
    (time, state, _, _, _, _, _, _, _, cm) = simulate(world,
                                                      initial_state=_initial_state(),
                                                      vehicle=Multirotor(quad_params),
                                                      controller=SE3Control(quad_params),
                                                      trajectory=HoverTraj(x0=[0.5, 0.5, -1.0]),
                                                      wind_profile=NoWind(),
                                                      imu=Imu(sampling_rate=sim_rate),
                                                      mocap=MotionCapture(sampling_rate=sim_rate),
                                                      estimator=NullEstimator(),
                                                      t_final=0.2,
                                                      t_step=1/sim_rate,
                                                      safety_margin=0.25,
                                                      use_mocap=False,
                                                      camera=cam)

    assert cm['visible_mask'].any(), "expected visible features in this setup"
    for i in range(cm['time'].shape[0]):
        direct = cam.measurement({'x': state['x'][i], 'q': state['q'][i]}, world)
        assert np.allclose(cm['image'][i].astype(float)/255.0, direct['image'], atol=1.01/255), \
            f"stored frame {i} should match a direct measurement()"
        assert np.array_equal(cm['visible_mask'][i], direct['visible_mask'])
        assert np.allclose(cm['depth'][i], direct['depth'], atol=1e-4)
        assert np.allclose(cm['projected'][i], direct['projected'], atol=1e-3)


def test_environment_camera_integration(tmp_path):
    """
    End-to-end Environment run with a camera: results contain measurements,
    plotting works, the csv excludes image data, and save_camera_data()
    round-trips to .npz + pngs.
    """
    print("\nTesting Environment camera integration")
    pandas = pytest.importorskip("pandas")
    from rotorpy.environments import Environment
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.vehicles.crazyflie_params import quad_params
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.trajectories.hover_traj import HoverTraj
    from rotorpy.utils.postprocessing import unpack_sim_data

    world = _feature_world()
    cam = _camera(frame_rate=20)
    sim_instance = Environment(vehicle=Multirotor(quad_params),
                               controller=SE3Control(quad_params),
                               trajectory=HoverTraj(x0=[0.5, 0.5, -1.0]),
                               sim_rate=100,
                               world=world,
                               camera=cam)
    sim_instance.vehicle.initial_state = _initial_state()

    results = sim_instance.run(t_final=0.25,
                               terminate=False,
                               plot=True,
                               plot_camera=True,
                               animate_bool=False,
                               verbose=False)

    cm = results['camera_measurements']
    assert cm is not None and cm['time'].shape[0] > 0
    assert set(cm.keys()) == {'time', 'image', 'visible_mask', 'projected', 'depth',
                              'keypoints', 'keypoint_depths', 'visible_features'}

    # The camera figure must show min(4, K) sampled frames + the visibility plot.
    fig = plt.figure('Camera Measurements vs Time')
    n_samples = min(4, cm['time'].shape[0])
    assert len([ax for ax in fig.axes if len(ax.images) > 0]) == n_samples

    # The visibility plot marks the sampled frame times with vertical lines.
    ax_vis = [a for a in fig.axes if a.get_xlabel() == 'time, s'][0]
    vlines = [ln for ln in ax_vis.get_lines()
              if len(ln.get_xdata()) == 2 and np.isclose(ln.get_xdata()[0], ln.get_xdata()[1])]
    assert len(vlines) == n_samples

    # The 3D path draws frustum (8 segments) + triad (3 arms) per sample.
    fig3d = plt.figure('3D Path')
    n_lines_3d = sum(len(ax.lines) for ax in fig3d.axes)
    assert n_lines_3d >= n_samples * (8 + 3) + 2

    # The csv must not contain any image data.
    csv_path = str(tmp_path / 'sim.csv')
    sim_instance.save_to_csv(csv_path)
    df = pandas.read_csv(csv_path, index_col=0)
    assert all('image' not in col for col in df.columns), "images must not end up in the csv"
    assert unpack_sim_data(results).shape[1] == df.shape[1]

    # Camera data saves to npz (+ pngs) and round-trips.
    npz_path = str(tmp_path / 'cam.npz')
    sim_instance.save_camera_data(npz_path, save_pngs=True)
    loaded = np.load(npz_path, allow_pickle=True)
    assert np.array_equal(loaded['image'], cm['image'])
    assert np.allclose(loaded['time'], cm['time'])
    assert len(list(loaded['keypoints'])) == cm['time'].shape[0]
    frames_dir = npz_path[:-len('.npz')] + '_frames'
    n_pngs = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    assert n_pngs == cm['time'].shape[0], "one png per captured frame"


def test_animate_with_camera_panel(tmp_path):
    """
    animate() with camera frames renders the camera panel into the saved video.
    With the camera object supplied, the frustum/triad are drawn as well.
    """
    print("\nTesting animation camera panel")
    pytest.importorskip("PIL")  # pillow writer for gif saving in the test
    from rotorpy.simulate import simulate
    from rotorpy.utils.animate import animate
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.vehicles.crazyflie_params import quad_params
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.trajectories.hover_traj import HoverTraj
    from rotorpy.wind.default_winds import NoWind
    from rotorpy.sensors.imu import Imu
    from rotorpy.sensors.external_mocap import MotionCapture
    from rotorpy.estimators.nullestimator import NullEstimator
    from scipy.spatial.transform import Rotation

    world = _feature_world()
    cam = _camera(frame_rate=None)
    (time, state, _, _, _, _, _, _, _, cm) = simulate(world,
                                                      initial_state=_initial_state(),
                                                      vehicle=Multirotor(quad_params),
                                                      controller=SE3Control(quad_params),
                                                      trajectory=HoverTraj(x0=[0.5, 0.5, -1.0]),
                                                      wind_profile=NoWind(),
                                                      imu=Imu(sampling_rate=100),
                                                      mocap=MotionCapture(sampling_rate=100),
                                                      estimator=NullEstimator(),
                                                      t_final=0.15,
                                                      t_step=0.01,
                                                      safety_margin=0.25,
                                                      use_mocap=False,
                                                      camera=cam)

    R = Rotation.from_quat(state['q']).as_matrix()
    ani = animate(time, state['x'], R, state['wind'],
                  animate_wind=False, world=world, filename=None,
                  close_on_finish=False, camera_images=cm['image'],
                  camera_times=cm['time'], camera=cam)
    gif_path = str(tmp_path / 'anim_cam.gif')
    ani.save(gif_path, writer='pillow', fps=5)
    assert os.path.exists(gif_path) and os.path.getsize(gif_path) > 0
    plt.close('all')

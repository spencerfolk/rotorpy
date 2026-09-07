import numpy as np
import torch

import rotorpy
from rotorpy.simulate import ExitStatus, simulate_batch, merge_batch_camera_measurements, safety_exit_batch
from rotorpy.wind.default_winds import BatchedNoWind
from rotorpy.sensors.imu import BatchedImu
from rotorpy.sensors.camera import BatchedPinholeCamera
from rotorpy.trajectories.minsnap import BatchedMinSnap
from rotorpy.trajectories.hover_traj import BatchedHoverTraj
from rotorpy.vehicles.multirotor import Multirotor, BatchedMultirotorParams, BatchedMultirotor
from rotorpy.controllers.quadrotor_control import BatchedSE3Control, SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params as cf_quad_params
from rotorpy.vehicles.hummingbird_params import quad_params as hb_quad_params
from rotorpy.utils.trajgen_utils import generate_random_minsnap_traj
from rotorpy.world import World

def test_batched_operators():
    np.random.seed(10)
    num_drones = 10
    device = torch.device("cpu")
    init_rotor_speed = 1788.53
    # Set to 0 if you want sim results to be more deterministic (default value is 100)
    cf_quad_params["motor_noise_std"] = 0
    hb_quad_params["motor_noise_std"] = 0

    # We'll simulate half crazyflies, half hummingbirds
    all_quad_params = [cf_quad_params]*(num_drones//2) + [hb_quad_params]*(num_drones//2)

    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    batch_state = {'x': torch.randn(num_drones,3, device=device).double(),
                  'v': torch.randn(num_drones, 3, device=device).double(),
                  'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_drones, 1).double(),
                  'w': torch.randn(num_drones, 3, device=device).double(),
                  'wind': torch.zeros(num_drones, 3, device=device).double(),
                  'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_drones, 1).double()}
    trajectories = [generate_random_minsnap_traj(world, 3, 1.0, 1.0, 2.0, np.random.randn(3)) for _ in range(num_drones)]
    batch_minsnap = BatchedMinSnap(trajectories, device)
    ts = np.array([np.random.uniform(0, 1) for i in range(num_drones)])
    batch_flat_output= batch_minsnap.update(ts)

    # Make sure that BatchedMinSnap does the same thing as MinSnap
    for j, traj in enumerate(trajectories):
        seq_flat_output = traj.update(ts[j])
        for key in batch_flat_output.keys():
            assert np.all(np.abs(batch_flat_output[key][j].cpu().numpy() - seq_flat_output[key]) < 1e-3)

    batch_params = BatchedMultirotorParams(all_quad_params, num_drones, device)
    batched_ctrlr = BatchedSE3Control(batch_params, num_drones, device)
    batch_control_inputs = batched_ctrlr.update(0, batch_state, batch_flat_output)

    # Make sure that BatchedSE3Control does the same thing as SE3Control
    # Make sure that BatchedMultirotor does the same thing as MultiRotor
    control_abstractions = ["cmd_motor_speeds", "cmd_motor_thrusts", "cmd_ctbm", "cmd_ctbr",
                            "cmd_ctatt", "cmd_vel", "cmd_acc"]
    for j in range(num_drones):
        single_ctrlr = SE3Control(all_quad_params[j])
        seq_state = {key: batch_state[key][j].cpu().numpy() for key in batch_state.keys()}
        flat_output = {key: batch_flat_output[key][j].cpu().numpy() for key in batch_flat_output.keys()}
        seq_control_input = single_ctrlr.update(0, seq_state, flat_output)
        for key in batch_control_inputs.keys():
            assert np.all(np.abs(batch_control_inputs[key][j].cpu().numpy() - seq_control_input[key]) < 1e-3)

    for abstraction in control_abstractions:
        print(f"Testing control abstraction = {abstraction}")
        batched_multirotor = BatchedMultirotor(batch_params, num_drones, batch_state, device, control_abstraction=abstraction)
        batch_next_state = batched_multirotor.step(batch_state, batch_control_inputs, 0.01)
        for j in range(num_drones):
            single_ctrlr = SE3Control(all_quad_params[j])
            seq_state = {key: batch_state[key][j].cpu().numpy() for key in batch_state.keys()}
            flat_output = {key: batch_flat_output[key][j].cpu().numpy() for key in batch_flat_output.keys()}
            seq_control_input = single_ctrlr.update(0, seq_state, flat_output)
            single_multirotor = Multirotor(all_quad_params[j], control_abstraction=abstraction)
            seq_next_state = single_multirotor.step(seq_state, seq_control_input, 0.01)
            for key in batch_next_state.keys():
                # since rotor speeds are large, we need a higher tolerance here.
                if key == "rotor_speeds":
                    assert np.all(np.abs(batch_next_state[key][j].cpu().numpy() - seq_next_state[key]) < 1)
                else:
                    assert np.all(np.abs(batch_next_state[key][j].cpu().numpy() - seq_next_state[key]) < 5e-2)


if __name__ == "__main__":
    test_batched_operators()


def _never_exit(t, state):
    return np.zeros(state['x'].shape[0], dtype=bool)


def _hover_pursuit_batch(num_drones, xs, device="cpu"):
    device = torch.device(device)
    xs = np.asarray(xs, dtype=np.float64)
    x0 = {'x': torch.tensor(xs, dtype=torch.double),
          'v': torch.zeros(num_drones, 3, dtype=torch.double),
          'q': torch.tensor([0, 0, 0, 1], dtype=torch.double).repeat(num_drones, 1),
          'w': torch.zeros(num_drones, 3, dtype=torch.double),
          'wind': torch.zeros(num_drones, 3, dtype=torch.double),
          'rotor_speeds': torch.tensor([1788.53]*4, dtype=torch.double).repeat(num_drones, 1)}
    bp = BatchedMultirotorParams([dict(cf_quad_params)] * num_drones, num_drones, device)
    vehicle = BatchedMultirotor(bp, num_drones, x0, device=device, integrator='rk4',
                                control_abstraction='cmd_vel')
    targets = torch.tensor(xs, dtype=torch.double)

    class Pursuit:
        def update(self, t, state, flat, idxs=None):
            v = torch.clamp(2.0 * (targets - state['x']), -2.0, 2.0).double()
            v[:, 2] = torch.clamp(v[:, 2], -1.0, 1.0)
            return {'cmd_v': v}

    traj = BatchedHoverTraj(num_drones, x0=targets, device=device)
    wind = BatchedNoWind(num_drones)
    imu = BatchedImu(num_drones, device=device)
    return x0, vehicle, Pursuit(), traj, wind, imu


def test_simulate_batch_collision_shared_world():
    np.random.seed(0)
    world = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]},
                   "blocks": [{"extents": [0, 1, 0, 1, -1, 2], "color": [1, 0, 0]}]})
    num_drones = 3
    xs = [[0.5, 0.5, 0.5], [3.0, 3.0, 1.0], [6.0, 6.0, 1.0]]
    x0, vehicle, controller, traj, wind, imu = _hover_pursuit_batch(num_drones, xs)
    _, _, _, _, _, _, exit_status, exit_timesteps, _ = simulate_batch(
        world, x0, vehicle, controller, traj, wind, imu,
        np.full(num_drones, 0.5), 0.02, 0.25, terminate=_never_exit)
    assert exit_status[0] == ExitStatus.COLLISION
    assert exit_status[1] == ExitStatus.TIMEOUT
    assert exit_status[2] == ExitStatus.TIMEOUT
    assert exit_timesteps[0] == 1


def test_simulate_batch_collision_per_drone_worlds():
    np.random.seed(0)
    world_a = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]},
                     "blocks": [{"extents": [0, 1, 0, 1, -1, 2], "color": [1, 0, 0]}]})
    world_b = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]}, "blocks": []})
    num_drones = 2
    xs = [[0.9, 0.9, 0.5], [3.0, 3.0, 1.0]]
    x0, vehicle, controller, traj, wind, imu = _hover_pursuit_batch(num_drones, xs)
    _, _, _, _, _, _, exit_status, exit_timesteps, _ = simulate_batch(
        [world_a, world_b], x0, vehicle, controller, traj, wind, imu,
        np.full(num_drones, 0.5), 0.02, 0.25, terminate=_never_exit)
    assert exit_status[0] == ExitStatus.COLLISION
    assert exit_status[1] == ExitStatus.TIMEOUT
    assert exit_timesteps[0] == 1


def test_simulate_batch_collision_check_disabled():
    np.random.seed(0)
    world = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]},
                   "blocks": [{"extents": [0, 1, 0, 1, -1, 2], "color": [1, 0, 0]}]})
    num_drones = 1
    xs = [[0.5, 0.5, 0.5]]
    x0, vehicle, controller, traj, wind, imu = _hover_pursuit_batch(num_drones, xs)
    _, _, _, _, _, _, exit_status, exit_timesteps, _ = simulate_batch(
        world, x0, vehicle, controller, traj, wind, imu,
        np.full(num_drones, 0.5), 0.02, 0.25, terminate=_never_exit, check_collisions=False)
    assert exit_status[0] == ExitStatus.TIMEOUT
    assert exit_timesteps[0] > 1


def test_simulate_batch_terminate_false():
    """terminate=False must never terminate the batch early (regression: the
    loop crashed on `done | None`)."""
    np.random.seed(0)
    world = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]}, "blocks": []})
    num_drones = 2
    xs = [[3.0, 3.0, 1.0], [6.0, 6.0, 1.0]]
    x0, vehicle, controller, traj, wind, imu = _hover_pursuit_batch(num_drones, xs)
    _, _, _, _, _, _, exit_status, exit_timesteps, _ = simulate_batch(
        world, x0, vehicle, controller, traj, wind, imu,
        np.full(num_drones, 0.2), 0.02, 0.25, terminate=False)
    assert all(status == ExitStatus.TIMEOUT for status in exit_status)
    assert np.all(exit_timesteps == exit_timesteps[0])
    assert 11 <= exit_timesteps[0] <= 12


def test_safety_exit_batch_collision_matches_numpy():
    """The torch collision path must agree with the numpy World.collisions for
    both shared and per-drone worlds, and must not flag NaN (ended) positions."""
    np.random.seed(0)
    block = {"extents": [0, 1, 0, 1, -1, 2], "color": [1, 0, 0]}
    world_a = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]}, "blocks": [block]})
    world_b = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]}, "blocks": []})
    margin = 0.25

    xs = np.vstack([
        [0.5, 0.5, 0.5],    # inside the block
        [1.05, 0.5, 0.5],   # within margin of the block
        [5.0, 5.0, 1.0],    # clear
        [-6.0, 0.0, 1.0],   # outside the world boundary
        [np.nan, np.nan, np.nan],  # ended drone: must not collide
    ])

    state = {'x': torch.tensor(xs, dtype=torch.double),
             'v': torch.zeros(len(xs), 3, dtype=torch.double),
             'w': torch.zeros(len(xs), 3, dtype=torch.double)}
    expected_shared = world_a.collisions(xs, margin)
    for check in (True, False):
        _, ce = safety_exit_batch(world_a, margin, state, None, None, check_collisions=check)
        if check:
            assert np.array_equal(ce, expected_shared)
        else:
            assert not np.any(ce)

    worlds = [world_a, world_a, world_a, world_a, None]
    expected_per_drone = np.array([worlds[b].collisions(xs[b].reshape(1, 3), margin)[0]
                                   if worlds[b] is not None else False
                                   for b in range(len(xs))])
    _, ce = safety_exit_batch(worlds, margin, state, None, None, check_collisions=True)
    assert np.array_equal(ce, expected_per_drone)


def test_simulate_batch_camera_capture_shared_world():
    torch.set_num_threads(1)
    np.random.seed(0)
    world = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]},
                   "blocks": [{"extents": [3, 4, 3, 4, -1, 4], "color": [1, 0, 0]}]})
    num_drones = 2
    xs = [[0.5, 0.5, 0.5], [1.5, 1.5, 0.5]]
    x0, vehicle, controller, traj, wind, imu = _hover_pursuit_batch(num_drones, xs)
    cam = BatchedPinholeCamera(
        num_drones, intrinsics={'fx': 100, 'fy': 100, 'width': 32, 'height': 32,
                                'cx': 16, 'cy': 16, 'dist_coeffs': np.zeros(5)},
        frame_rate=10)
    _, _, _, _, _, _, _, _, camera_measurements = simulate_batch(
        world, x0, vehicle, controller, traj, wind, imu,
        np.full(num_drones, 0.35), 0.1, 0.25, terminate=_never_exit, camera=cam)
    assert camera_measurements is not None
    K = camera_measurements['time'].shape[0]
    assert np.all(np.abs(np.diff(camera_measurements['time']) - 0.1) < 1e-6)
    assert camera_measurements['image'].shape == (K, num_drones, 32, 32, 3)
    assert camera_measurements['image'].dtype == np.uint8
    assert camera_measurements['running'].shape == (K, num_drones)
    assert np.all(camera_measurements['running'][0])


def test_simulate_batch_camera_capture_per_drone_worlds():
    torch.set_num_threads(1)
    np.random.seed(0)
    world_a = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]},
                     "blocks": [{"extents": [3, 4, 3, 4, -1, 4], "color": [1, 0, 0]}]})
    world_b = World({"bounds": {"extents": [-5, 10, -5, 10, -1, 10]}, "blocks": []})
    num_drones = 2
    xs = [[0.5, 0.5, 0.5], [1.5, 1.5, 0.5]]
    x0, vehicle, controller, traj, wind, imu = _hover_pursuit_batch(num_drones, xs)
    cam = BatchedPinholeCamera(
        num_drones, intrinsics={'fx': 100, 'fy': 100, 'width': 32, 'height': 32,
                                'cx': 16, 'cy': 16, 'dist_coeffs': np.zeros(5)},
        frame_rate=10)
    _, _, _, _, _, _, _, _, camera_measurements = simulate_batch(
        [world_a, world_b], x0, vehicle, controller, traj, wind, imu,
        np.full(num_drones, 0.35), 0.1, 0.25, terminate=_never_exit, camera=cam)
    K = camera_measurements['time'].shape[0]
    assert camera_measurements['image'].shape == (K, num_drones, 32, 32, 3)
    assert camera_measurements['visible_mask'].shape == (K, num_drones)
    assert camera_measurements['keypoints'].shape == (K, num_drones)
    assert camera_measurements['visible_mask'][0, 0].shape == (0,)
    assert camera_measurements['visible_mask'][0, 0].dtype == bool


def test_merge_batch_camera_measurements_shared():
    num_drones, N, D = 3, 4, 6

    def fake_meas(run_idx):
        R = len(run_idx)
        return {
            'image': torch.rand(R, 8, 8, 3),
            'visible_mask': torch.ones(R, N, dtype=torch.bool),
            'depth': torch.rand(R, N),
            'projected': torch.rand(R, N, 2),
            'colors': torch.rand(R, N, 3),
            'descriptors': torch.rand(R, N, D),
            'keypoints': [torch.rand(3, 2) for _ in range(R)],
            'keypoint_depths': [torch.rand(3) for _ in range(R)],
            'visible_features': [torch.rand(3, 3) for _ in range(R)],
            'visible_colors': [torch.rand(3, 3) for _ in range(R)],
            'visible_descriptors': [torch.rand(3, D) for _ in range(R)],
        }

    frames = [(fake_meas([0, 1, 2]), np.array([0, 1, 2])),
              (fake_meas([0]), np.array([0]))]
    times = [0.0, 0.1]
    out = merge_batch_camera_measurements(frames, times, num_drones)
    assert out['image'].shape == (2, num_drones, 8, 8, 3)
    assert out['image'].dtype == np.uint8
    assert out['visible_mask'].shape == (2, num_drones, N)
    assert out['visible_mask'][1, 1].sum() == 0
    assert out['projected'].shape == (2, num_drones, N, 2)
    assert np.all(np.isnan(out['projected'][1, 1]))
    assert out['depth'].shape == (2, num_drones, N)
    assert np.all(np.isnan(out['depth'][1, 2]))
    assert out['colors'].shape == (2, num_drones, N, 3)
    assert out['descriptors'].shape == (2, num_drones, N, D)
    assert out['keypoints'].shape == (2, num_drones)
    assert out['keypoints'][1, 2] is None
    assert out['keypoints'][0, 0].shape == (3, 2)
    assert np.all(out['running'][0]) and not np.any(out['running'][1, 1:])

    frames_rgb = [(dict(fake_meas([0, 1, 2]), descriptors=None, visible_descriptors=None),
                   np.array([0, 1, 2]))]
    out = merge_batch_camera_measurements(frames_rgb, [0.0], num_drones)
    assert out['descriptors'] is None
    assert out['visible_descriptors'] is None
    assert out['colors'].shape == (1, num_drones, N, 3)


def test_merge_batch_camera_measurements_per_drone():
    D = 6

    def fake_meas(entries):
        cols, descs = [], []
        for n_b, _ in entries:
            cols.append(torch.rand(n_b, 3) if n_b else torch.empty(0, 3))
            descs.append(torch.rand(n_b, D) if n_b else None)
        return {
            'image': torch.rand(len(entries), 8, 8, 3),
            'visible_mask': [torch.ones(n, dtype=torch.bool) for n, _ in entries],
            'depth': [torch.rand(n) for n, _ in entries],
            'projected': [torch.rand(n, 2) for n, _ in entries],
            'colors': cols,
            'descriptors': descs,
            'keypoints': [torch.rand(m, 2) for _, m in entries],
            'keypoint_depths': [torch.rand(m) for _, m in entries],
            'visible_features': [torch.rand(m, 3) for _, m in entries],
            'visible_colors': [torch.rand(m, 3) for _, m in entries],
            'visible_descriptors': [torch.rand(m, D) for _, m in entries],
        }

    frames = [(fake_meas([(3, 1), (2, 0)]), np.array([0, 1])),
              (fake_meas([(4, 2), (0, 0)]), np.array([2, 1]))]
    times = [0.0, 0.1]
    out = merge_batch_camera_measurements(frames, times, 3)
    assert out['image'].shape == (2, 3, 8, 8, 3)
    assert out['visible_mask'].shape == (2, 3)
    assert out['visible_mask'][0, 1].shape == (2,)
    assert out['visible_mask'][1, 1].shape == (0,)
    assert out['visible_mask'][0, 2] is None
    assert out['projected'].shape == (2, 3)
    assert out['projected'][0, 0].shape == (3, 2)
    assert out['projected'][1, 2].shape == (4, 2)
    assert out['depth'][0, 2] is None
    assert out['colors'][0, 0].shape == (3, 3)
    assert out['descriptors'][1, 2].shape == (4, D)
    assert out['keypoints'][1, 2].shape == (2, 2)
    assert out['keypoints'][1, 0] is None

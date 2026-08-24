'''
Tests for the pinhole camera sensor and world visual features.
'''

import numpy as np
import pytest


def _block_world(blocks, bounds=None, add_features=True, feature_mode='regular',
                 feature_spacing=0.5, N_features_per_surface=100, seed=None,
                 descriptor_noise=0.05):
    """
    Build a World from a list of block dicts (extents + color).
    """
    from rotorpy.world import World
    if bounds is None:
        bounds = {'extents': [-2, 2, -2, 2, -2, 2]}
    world_data = {'bounds': bounds, 'blocks': blocks}
    return World(world_data, add_features=add_features, feature_mode=feature_mode,
                 feature_spacing=feature_spacing, N_features_per_surface=N_features_per_surface,
                 seed=seed, descriptor_noise=descriptor_noise)


def _camera_looking_along_x(extrinsics_position=None):
    """
    A PinholeCamera whose body->camera rotation maps world +x to camera +z
    (camera looks along the world +x axis when the vehicle is yaw-aligned
    with the world frame).
    """
    from rotorpy.sensors.camera import PinholeCamera
    from scipy.spatial.transform import Rotation
    if extrinsics_position is None:
        extrinsics_position = np.zeros(3)
    return PinholeCamera(extrinsics={'position': np.array(extrinsics_position, dtype=np.float64),
                                     'orientation': Rotation.from_euler('y', -90, degrees=True).as_quat()})


def test_feature_generation_regular():
    print("\nTesting regular surface feature generation")
    from rotorpy.world import World

    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [-1, 1, -1, 1, 0, 2], 'color': [1, 0, 0]}],
    }
    world = World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.5)
    features = world.get_surface_features()
    assert features is not None, "expected surface features to be generated"
    assert features.ndim == 2 and features.shape[1] == 3
    assert features.shape[0] > 0, "expected at least one feature"

    # Every feature must lie on some block face: at least one coordinate must
    # coincide with a face plane of the block.
    x_planes = np.array([-1.0, 1.0])
    y_planes = np.array([-1.0, 1.0])
    z_planes = np.array([0.0, 2.0])
    for f in features:
        on_face = (np.any(np.abs(f[0] - x_planes) < 1e-6) or
                   np.any(np.abs(f[1] - y_planes) < 1e-6) or
                   np.any(np.abs(f[2] - z_planes) < 1e-6))
        assert on_face, f"feature {f} does not lie on a block face"

    # Descriptors match the feature count and live in [0, 1].
    descriptors = world.get_feature_descriptors()
    assert descriptors is not None, "expected feature descriptors to be generated"
    assert descriptors.shape == features.shape
    assert np.all(descriptors >= 0.0) and np.all(descriptors <= 1.0)


def test_feature_generation_random_seeded():
    print("\nTesting seeded random surface feature generation")
    blocks = [{'extents': [-1, 1, -1, 1, 0, 2], 'color': [1, 0, 0]}]
    w1 = _block_world(blocks, feature_mode='random', N_features_per_surface=50, seed=42)
    w2 = _block_world(blocks, feature_mode='random', N_features_per_surface=50, seed=42)
    f1, f2 = w1.get_surface_features(), w2.get_surface_features()
    d1, d2 = w1.get_feature_descriptors(), w2.get_feature_descriptors()
    assert f1 is not None and f2 is not None
    assert d1 is not None and d2 is not None
    assert f1.shape[0] > 0
    assert np.array_equal(f1, f2), "same seed should give identical features"
    assert np.array_equal(d1, d2), "same seed should give identical descriptors"


def test_feature_descriptors_no_jitter():
    print("\nTesting zero descriptor noise")
    blocks = [{'extents': [-1, 1, -1, 1, 0, 2], 'color': [1, 0, 0]}]
    world = _block_world(blocks, feature_mode='regular', feature_spacing=0.5, descriptor_noise=0.0)
    descriptors = world.get_feature_descriptors()
    assert descriptors is not None
    assert descriptors.shape[0] > 0
    assert np.allclose(descriptors, np.array([1.0, 0.0, 0.0]), atol=1e-9)


def test_pinhole_projection():
    print("\nTesting PinholeCamera projection")
    from rotorpy.sensors.camera import PinholeCamera
    from scipy.spatial.transform import Rotation

    intrinsics = {'fx': 500., 'fy': 400., 'width': 640, 'height': 480, 'cx': 320., 'cy': 240., 'dist_coeffs': np.zeros(5)}
    cam = PinholeCamera(intrinsics=intrinsics)
    pts_cam = np.array([[1.0, 0.5, 2.0]])
    pixels = cam.project_points(pts_cam)
    assert np.allclose(pixels, np.array([[570.0, 340.0]]))

    # 90 deg yaw about z extrinsics (body -> camera), identity vehicle state.
    q_BC = np.array([0.0, 0.0, 0.7071068, 0.7071068])
    cam2 = PinholeCamera(extrinsics={'position': np.zeros(3), 'orientation': q_BC})
    state = {'x': np.zeros(3), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    pose = cam2.compute_camera_pose(state)
    p_W = np.array([1.0, 0.5, 2.0])

    # Manual computation: R_WC should be a 90 deg rotation about z.
    R_WC = Rotation.from_quat(pose['q']).as_matrix()
    assert np.allclose(R_WC, Rotation.from_euler('z', 90, degrees=True).as_matrix(), atol=1e-6)
    p_c = R_WC @ p_W
    assert np.allclose(cam2.world_to_camera(p_W.reshape(1, 3), pose)[0], p_c)
    manual = np.array([cam2.fx * p_c[0] / p_c[2] + cam2.cx,
                       cam2.fy * p_c[1] / p_c[2] + cam2.cy])
    assert np.allclose(cam2.project_points(p_c.reshape(1, 3))[0], manual)


def test_distortion_shifts_projection():
    print("\nTesting distortion effect on projection")
    from rotorpy.sensors.camera import PinholeCamera

    intrinsics = {'fx': 500., 'fy': 400., 'width': 640, 'height': 480, 'cx': 320., 'cy': 240., 'dist_coeffs': np.zeros(5)}
    cam_undistorted = PinholeCamera(intrinsics=intrinsics)
    intrinsics_k1 = dict(intrinsics)
    intrinsics_k1['dist_coeffs'] = np.array([0.1, 0.0, 0.0, 0.0, 0.0])
    cam_distorted = PinholeCamera(intrinsics=intrinsics_k1)

    pts_cam = np.array([[1.0, 0.5, 2.0]])
    p_undistorted = cam_undistorted.project_points(pts_cam)
    p_distorted = cam_distorted.project_points(pts_cam)
    assert not np.allclose(p_undistorted, p_distorted), "k1 != 0 should move the projection"
    assert np.linalg.norm(p_distorted - p_undistorted) > 1.0


def test_extrinsics_offset():
    print("\nTesting extrinsics position offset")
    from rotorpy.sensors.camera import PinholeCamera

    identity = {'position': np.zeros(3), 'orientation': np.array([0.0, 0.0, 0.0, 1.0])}
    cam_origin = PinholeCamera(extrinsics=identity)
    cam_offset = PinholeCamera(extrinsics={'position': np.array([0.5, 0.0, 0.0]),
                                           'orientation': np.array([0.0, 0.0, 0.0, 1.0])})
    state = {'x': np.zeros(3), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    pts_world = np.array([[1.0, 0.5, 2.0]])

    p_origin = cam_origin.project_points(cam_origin.world_to_camera(pts_world, cam_origin.compute_camera_pose(state)))
    p_offset = cam_offset.project_points(cam_offset.world_to_camera(pts_world, cam_offset.compute_camera_pose(state)))
    assert not np.allclose(p_origin, p_offset), "extrinsics position offset should shift the projection"
    assert np.linalg.norm(p_offset - p_origin) > 1.0


def test_occlusion_visibility():
    print("\nTesting occlusion visibility")
    from rotorpy.world import World

    bounds = {'extents': [-3, 2, -1, 1, -1, 1]}
    # Block A in front (closer to the camera), block B behind it sharing the face at x=0.5.
    world_data = {
        'bounds': bounds,
        'blocks': [
            {'extents': [0.0, 0.5, 0.0, 0.5, 0.0, 0.5], 'color': [1, 0, 0]},
            {'extents': [0.5, 1.0, 0.0, 0.5, 0.0, 0.5], 'color': [0, 1, 0]},
        ],
    }
    world = World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.25, descriptor_noise=0.0)
    features = world.get_surface_features()
    assert features is not None and features.shape[0] > 0

    # Camera at x=-2 looking along +x, centered on y=z=0.25.
    cam = _camera_looking_along_x()
    state = {'x': np.array([-2.0, 0.25, 0.25]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    pose = cam.compute_camera_pose(state)

    # Features on block A's near face (x=0) must be visible.
    near_A = features[np.abs(features[:, 0] - 0.0) < 1e-6]
    assert near_A.shape[0] > 0
    assert np.all(cam.compute_occlusion(near_A, pose, world)), "near face of block A should be visible"

    # Features on block B's near face (x=0.5) must be occluded by block A.
    near_B = features[np.abs(features[:, 0] - 0.5) < 1e-6]
    assert near_B.shape[0] > 0
    assert not np.any(cam.compute_occlusion(near_B, pose, world)), "near face of block B should be occluded by block A"

    # A feature on the FAR face of block A (x=0.5) is occluded by A itself,
    # even with no block behind it.
    world_a = World({'bounds': bounds,
                     'blocks': [{'extents': [0.0, 0.5, 0.0, 0.5, 0.0, 0.5], 'color': [1, 0, 0]}]},
                    add_features=True, feature_mode='regular', feature_spacing=0.25, descriptor_noise=0.0)
    far_A = world_a.get_surface_features()
    far_A = far_A[np.abs(far_A[:, 0] - 0.5) < 1e-6]
    assert far_A.shape[0] > 0
    assert not np.any(cam.compute_occlusion(far_A, pose, world_a)), "far face of block A should be occluded by block A itself"

    # A point in front of block A (on no block) is visible.
    feat_front = np.array([[-0.25, 0.25, 0.25]])
    assert cam.compute_occlusion(feat_front, pose, world)[0], "point in front of block A should be visible"


def test_occlusion_missed_box_not_occluded():
    print("\nTesting occlusion: a ray that misses a box is not occluded")
    from rotorpy.world import World

    # Regression test: a ray that passes *beside* a small box used to be
    # flagged as occluded because the slab test was missing the t_entry <=
    # t_exit intersection check (t_entry > t_exit means the ray misses).
    bounds = {'extents': [0, 6, 0, 4, 0, 3]}
    world_data = {
        'bounds': bounds,
        'blocks': [
            # Small near box offset up+right of the camera's line of sight.
            {'extents': [2.2, 2.6, 1.5, 1.9, 0.9, 1.3], 'color': [1, 0, 0]},
            {'extents': [3.6, 5.6, 0.7, 2.5, 0.3, 2.1], 'color': [0, 0, 1]},
        ],
    }
    world = World(world_data, add_features=True, feature_mode='regular',
                  feature_spacing=0.1, descriptor_noise=0.0)
    cam = _camera_looking_along_x()
    state = {'x': np.array([0.5, 1.6, 1.2]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    pose = cam.compute_camera_pose(state)

    # A point on the far cuboid's near face, left of the near box's projection:
    # the ray to it misses the near box entirely, so it must be visible.
    clear_point = np.array([[3.6, 0.8, 2.0]])
    assert cam.compute_occlusion(clear_point, pose, world)[0], \
        "point beside the near box should not be occluded"

    # A point on the near face that lies directly behind the near box: the ray
    # passes through the box, so it must be occluded.
    behind_point = np.array([[3.6, 2.0, 1.2]])
    assert not cam.compute_occlusion(behind_point, pose, world)[0], \
        "point directly behind the near box should be occluded"

    # Far cuboid features must be at least partially visible (the rim).
    features = world.get_surface_features()
    on_B = features[:, 0] >= 3.6
    vis = cam.compute_occlusion(features, pose, world)
    assert vis[on_B].any(), "some far cuboid features must be visible around the near box"


def test_render_output():
    print("\nTesting render output")
    from rotorpy.world import World

    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 'color': [1.0, 0.0, 0.0]}],
    }
    world = World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.25, descriptor_noise=0.0)
    empty_world = World.empty(extents=[-2, 2, -2, 2, -2, 2], add_features=True)

    cam = _camera_looking_along_x()
    state = {'x': np.array([-1.0, 0.5, 0.5]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    out = cam.render(world, state)

    image = out['image']
    assert image.shape == (cam.height, cam.width, 3)
    assert np.issubdtype(image.dtype, np.floating)
    assert np.all(image >= 0.0) and np.all(image <= 1.0)

    N = world.get_surface_features().shape[0]
    assert out['visible_mask'].shape == (N,)
    assert out['projected'].shape == (N, 2)
    assert out['depth'].shape == (N,)
    M = out['visible_features'].shape[0]
    assert M == out['keypoints'].shape[0] == out['keypoint_depths'].shape[0]
    assert M > 0, "expected at least one visible feature"

    # The rendered image must differ from a render of an empty world.
    empty_out = cam.render(empty_world, state)
    assert not np.allclose(image, empty_out['image']), "visible features should change the image"

    # Every visible feature pixel must not be the background color.
    background = np.array([0.9, 0.9, 0.9])
    for kp in out['keypoints']:
        px = image[int(round(kp[1])), int(round(kp[0]))]
        assert not np.allclose(px, background, atol=1e-6), "visible feature pixel should not be the background"


def test_batched_camera_matches_numpy():
    print("\nTesting batched camera vs numpy camera")
    torch = pytest.importorskip("torch")
    from rotorpy.world import World
    from rotorpy.sensors.camera import PinholeCamera, BatchedPinholeCamera

    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 'color': [1.0, 0.0, 0.0]}],
    }
    world = World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.25,
                  descriptor_noise=0.1, seed=0)

    extrinsics = {'position': np.zeros(3),
                  'orientation': np.array([0.0, 0.0, 0.0, 1.0])}
    cam = PinholeCamera(extrinsics=extrinsics)
    bcam = BatchedPinholeCamera(num_drones=2, extrinsics=extrinsics)

    # One drone at identity pose, one offset. Camera (identity extrinsics) looks
    # along +z, so place the drones below the block.
    state0 = {'x': np.array([0.5, 0.5, -1.0]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    state1 = {'x': np.array([0.25, 0.5, -1.5]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}

    out0 = cam.render(world, state0)
    out1 = cam.render(world, state1)

    states = {'x': torch.tensor(np.stack([state0['x'], state1['x']])),
              'q': torch.tensor(np.stack([state0['q'], state1['q']]))}
    bout = bcam.render(world, states)
    assert tuple(bout['image'].shape) == (2, cam.height, cam.width, 3)

    for b, out in enumerate([out0, out1]):
        assert torch.allclose(bout['image'][b], torch.from_numpy(out['image']), atol=1e-4), f"image mismatch drone {b}"
        assert torch.equal(bout['visible_mask'][b], torch.from_numpy(out['visible_mask'])), f"visible_mask mismatch drone {b}"
        assert torch.allclose(bout['depth'][b], torch.from_numpy(out['depth']).float(), atol=1e-4), f"depth mismatch drone {b}"
        assert torch.allclose(bout['keypoints'][b], torch.from_numpy(out['keypoints']).float(), atol=1e-4), f"keypoints mismatch drone {b}"
        assert torch.allclose(bout['keypoint_depths'][b], torch.from_numpy(out['keypoint_depths']).float(), atol=1e-4), f"keypoint_depths mismatch drone {b}"
        assert torch.allclose(bout['visible_features'][b], torch.from_numpy(out['visible_features']).float(), atol=1e-4), f"visible_features mismatch drone {b}"


def test_empty_world_feature_consistency():
    print("\nTesting empty world feature consistency")
    from rotorpy.world import World

    world = World.empty(extents=[-2, 2, -2, 2, -2, 2], add_features=True)
    features = world.get_surface_features()
    descriptors = world.get_feature_descriptors()
    assert features is not None, "empty world should still return a (0, 3) feature array"
    assert features.shape == (0, 3)
    assert descriptors is not None, "empty world should still return a (0, 3) descriptor array"
    assert descriptors.shape == (0, 3)

    world_no_features = World.empty(extents=[-2, 2, -2, 2, -2, 2], add_features=False)
    assert world_no_features.get_surface_features() is None


def test_render_near_plane_culling():
    print("\nTesting near plane culling in render")
    from rotorpy.world import World
    from rotorpy.sensors.camera import PinholeCamera

    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 'color': [1.0, 0.0, 0.0]}],
    }
    world = World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.25, descriptor_noise=0.0)

    # Camera below the block, looking along +z. Bottom-face features (z=0) sit at
    # camera-frame depth 0.01.
    state = {'x': np.array([0.5, 0.5, -0.01]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    cam_default = PinholeCamera()
    cam_near_plane_0 = PinholeCamera(near_plane=0.0)

    out_default = cam_default.render(world, state)
    out_near0 = cam_near_plane_0.render(world, state)

    features = world.get_surface_features()
    depth = out_default['depth']
    near = depth < cam_default.near_plane
    assert np.any(near), "expected some features within the near plane for this setup"
    assert not np.any(out_default['visible_mask'][near]), \
        "features with camera-frame depth below near_plane must not be visible"
    assert np.any(out_near0['visible_mask'][near]), \
        "features within the near plane should become visible when near_plane is reduced to 0"


def test_render_nonidentity_attitude():
    print("\nTesting render with non-identity vehicle attitude")
    from rotorpy.world import World
    from rotorpy.sensors.camera import PinholeCamera

    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 'color': [1.0, 0.0, 0.0]}],
    }
    world = World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.25, descriptor_noise=0.0)

    cam = PinholeCamera()
    state_id = {'x': np.array([0.5, 0.5, -1.0]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    # 90 deg pitch about x: body +z (the optical axis) now points along world -y,
    # so the camera sees a different side of the block than the identity view.
    q_pitch = np.array([0.7071068, 0.0, 0.0, 0.7071068])
    state_pitch = {'x': np.array([0.5, 0.5, -1.0]), 'q': q_pitch}

    out_id = cam.render(world, state_id)
    out_pitch = cam.render(world, state_pitch)
    assert not np.allclose(out_id['image'], out_pitch['image']), \
        "rotating the vehicle attitude should change the rendered image"


def test_render_without_distortion_and_splat():
    print("\nTesting with_distortion=False and splat_radius")
    from rotorpy.world import World
    from rotorpy.sensors.camera import PinholeCamera

    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 'color': [1.0, 0.0, 0.0]}],
    }
    world = World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.25, descriptor_noise=0.0)

    intrinsics = {'fx': 500., 'fy': 500., 'width': 640, 'height': 480, 'cx': 320., 'cy': 240.,
                  'dist_coeffs': np.array([0.1, 0.0, 0.0, 0.0, 0.0])}
    cam = PinholeCamera(intrinsics=intrinsics)
    state = {'x': np.array([0.5, 0.5, -1.0]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}

    out_dist = cam.render(world, state, with_distortion=True)
    out_nodist = cam.render(world, state, with_distortion=False)
    assert not np.allclose(out_dist['image'], out_nodist['image']), \
        "distortion should alter the rendered image"

    out_r0 = cam.render(world, state, splat_radius=0)
    out_r2 = cam.render(world, state, splat_radius=2)
    background = np.array([0.9, 0.9, 0.9])
    n_nonbg_r0 = np.sum(np.linalg.norm(out_r0['image'] - background, axis=-1) > 1e-6)
    n_nonbg_r2 = np.sum(np.linalg.norm(out_r2['image'] - background, axis=-1) > 1e-6)
    assert n_nonbg_r2 > n_nonbg_r0, "larger splat radius should color more pixels"


def test_batched_camera_distortion_matches_numpy():
    print("\nTesting batched camera distortion vs numpy camera")
    torch = pytest.importorskip("torch")
    from rotorpy.world import World
    from rotorpy.sensors.camera import PinholeCamera, BatchedPinholeCamera

    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 'color': [1.0, 0.0, 0.0]}],
    }
    world = World(world_data, add_features=True, feature_mode='regular', feature_spacing=0.25,
                  descriptor_noise=0.1, seed=1)

    intrinsics = {'fx': 500., 'fy': 500., 'width': 640, 'height': 480, 'cx': 320., 'cy': 240.,
                  'dist_coeffs': np.array([0.1, -0.02, 0.001, 0.002, 0.0])}
    extrinsics = {'position': np.zeros(3), 'orientation': np.array([0.0, 0.0, 0.0, 1.0])}
    cam = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)
    bcam = BatchedPinholeCamera(num_drones=2, intrinsics=intrinsics, extrinsics=extrinsics)

    states = [{'x': np.array([0.5, 0.5, -1.0]), 'q': np.array([0.0, 0.0, 0.0, 1.0])},
              {'x': np.array([0.25, 0.5, -1.5]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}]
    outs = [cam.render(world, s) for s in states]
    bout = bcam.render(world, {'x': torch.tensor(np.stack([s['x'] for s in states])),
                               'q': torch.tensor(np.stack([s['q'] for s in states]))})

    for b, out in enumerate(outs):
        assert torch.allclose(bout['image'][b], torch.from_numpy(out['image']), atol=1e-4), f"image mismatch drone {b}"
        assert torch.equal(bout['visible_mask'][b], torch.from_numpy(out['visible_mask'])), f"visible_mask mismatch drone {b}"
        assert torch.allclose(bout['depth'][b], torch.from_numpy(out['depth']).float(), atol=1e-4), f"depth mismatch drone {b}"


def test_batched_camera_batch_sizes():
    print("\nTesting batched camera with B != num_drones")
    torch = pytest.importorskip("torch")
    from rotorpy.world import World
    from rotorpy.sensors.camera import BatchedPinholeCamera

    world = World.empty(extents=[-2, 2, -2, 2, -2, 2], add_features=True)
    bcam = BatchedPinholeCamera(num_drones=3)

    states = {'x': torch.zeros(3, 3), 'q': torch.zeros(3, 4)}
    states['q'][:, 3] = 1.0
    bout = bcam.render(world, states)
    assert bout['image'].shape[0] == 3

    # B < num_drones is allowed.
    states2 = {'x': torch.zeros(2, 3), 'q': torch.zeros(2, 4)}
    states2['q'][:, 3] = 1.0
    bout2 = bcam.render(world, states2)
    assert bout2['image'].shape[0] == 2

    # B > num_drones must raise.
    states3 = {'x': torch.zeros(4, 3), 'q': torch.zeros(4, 4)}
    states3['q'][:, 3] = 1.0
    with pytest.raises(ValueError):
        bcam.render(world, states3)


def test_camera_intrinsics_defaults():
    print("\nTesting camera intrinsics/extrinsics defaults")
    from rotorpy.sensors.camera import PinholeCamera

    cam = PinholeCamera()
    assert cam.fx == 500.0
    assert cam.fy == 500.0
    assert cam.width == 640
    assert cam.height == 480
    assert cam.cx == 320.0
    assert cam.cy == 240.0
    assert cam.dist_coeffs.shape == (5,)
    assert np.allclose(cam.dist_coeffs, 0.0)
    assert cam.p_BC.shape == (3,)
    assert cam.R_BC.shape == (3, 3)
    assert np.allclose(cam.p_BC, 0.0)
    assert np.allclose(cam.R_BC, np.eye(3))


def test_camera_triad_world_directions():
    print("\nTesting camera triad draws camera-frame axes in the world frame")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    from rotorpy.utils.camera_plotter import draw_camera_triad

    cam = _camera_looking_along_x()
    state = {'x': np.array([0.5, 1.6, 1.2]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    pose = cam.compute_camera_pose(state)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    artists = draw_camera_triad(ax, pose, scale=1.0)
    plt.close(fig)

    assert len(artists) == 3, "triad should draw three arms"
    colors = [artist.get_color() for artist in artists]
    assert colors == ['r', 'g', 'b'], "triad arms should be red (x), green (y), blue (z)"

    R_WC = Rotation.from_quat(pose['q']).as_matrix()
    for i, artist in enumerate(artists):
        (xs, ys, zs) = artist.get_data_3d()
        start = np.array([xs[0], ys[0], zs[0]])
        end = np.array([xs[1], ys[1], zs[1]])
        assert np.allclose(start, pose['x']), "triad should start at the camera origin"
        expected_end = pose['x'] + R_WC.T @ np.eye(3)[i]
        assert np.allclose(end, expected_end), \
            f"triad arm {i} should point along R_WC.T @ e_{i}"

    # The blue (z) arm points along the boresight: world +x for this camera.
    assert np.allclose(artists[2].get_data_3d()[0][1],
                       pose['x'][0] + 1.0, atol=1e-6)


def test_nonidentity_drone_rotation():
    """
    Verify camera pose and world_to_camera correctness when the drone has a
    non-identity rotation and the extrinsics are identity.

    With identity extrinsics the camera frame equals the body frame, so the
    world-to-camera rotation should equal the body-to-world rotation transposed
    (i.e. world-to-body).
    """
    from rotorpy.sensors.camera import PinholeCamera
    from scipy.spatial.transform import Rotation

    cam = PinholeCamera()  # identity extrinsics, camera looks along +z

    # 90 deg yaw about z: body +x -> world +y, body +y -> world -x.
    q_yaw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    state = {'x': np.array([1.0, 2.0, 3.0]), 'q': q_yaw}

    pose = cam.compute_camera_pose(state)

    # Camera position: p_WC = x + R_WB @ p_BC = x (since p_BC = 0).
    assert np.allclose(pose['x'], state['x']), "camera position should equal drone position"

    # World-to-camera rotation: R_WC = R_BC @ R_WB^T = R_WB^T (identity extrinsics).
    R_WC = Rotation.from_quat(pose['q']).as_matrix()
    R_WB = Rotation.from_quat(q_yaw).as_matrix()
    assert np.allclose(R_WC, R_WB.T), "R_WC should be R_WB^T for identity extrinsics"

    # world_to_camera: a point at (2, 1, 3) in world frame.
    # Relative to camera at (1, 2, 3): (1, -1, 0).
    # R_WB for 90 deg yaw = [[0,-1,0],[1,0,0],[0,0,1]]
    # R_WC = R_WB^T = [[0,1,0],[-1,0,0],[0,0,1]]
    # R_WC @ (1,-1,0) = (-1, -1, 0)
    p_W = np.array([[2.0, 1.0, 3.0]])
    p_c = cam.world_to_camera(p_W, pose)
    expected = R_WC @ (p_W[0] - pose['x'])
    assert np.allclose(p_c[0], expected), "world_to_camera should match manual R_WC @ (p - p_WC)"

    # Rendering: same geometry as test_render_nonidentity_attitude but verifying
    # the full pipeline (pose + world_to_camera + project + render) is correct
    # for a non-identity drone rotation. The math assertions above already
    # verified the pose and world_to_camera; this just confirms the rendered
    # image changes when the drone rotates.
    from rotorpy.world import World
    world_data = {
        'bounds': {'extents': [-2, 2, -2, 2, -2, 2]},
        'blocks': [{'extents': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 'color': [1.0, 0.0, 0.0]}],
    }
    # Use descriptor noise so features have distinct colors, ensuring
    # different views of the same cube face produce different images.
    world = World(world_data, add_features=True, feature_mode='regular',
                  feature_spacing=0.25, descriptor_noise=0.2)

    # Identity extrinsics camera (looks along +z). From below, looking up at
    # the block's bottom face.
    state_id = {'x': np.array([0.5, 0.5, -1.0]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    out_id = cam.render(world, state_id)
    assert out_id['visible_mask'].any(), "identity drone should see features"

    # 90 deg pitch about x: body +z -> world -y. Camera now looks along -y.
    # Place camera at y=2 looking down to see the y=1 face of the block.
    q_pitch = Rotation.from_euler('x', 90, degrees=True).as_quat()
    state_pitch = {'x': np.array([0.5, 2.0, 0.5]), 'q': q_pitch}
    out_pitch = cam.render(world, state_pitch)
    assert out_pitch['visible_mask'].any(), "pitched drone should see features"

    assert not np.allclose(out_id['image'], out_pitch['image']), \
        "different drone attitudes should produce different images"


def test_measurement_interface():
    """
    Verify that measurement() returns the same output as render().
    """
    from rotorpy.sensors.camera import PinholeCamera
    from rotorpy.world import World

    cam = PinholeCamera()
    world = World.empty((-2, 4, -2, 4, -1, 5), add_features=True,
                        feature_mode='regular', feature_spacing=0.5, descriptor_noise=0.0)
    state = {'x': np.array([1.0, 2.0, 0.0]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}

    out_render = cam.render(world, state)
    out_meas = cam.measurement(state, world)

    assert out_render.keys() == out_meas.keys(), "measurement() and render() should return the same keys"
    assert np.allclose(out_render['image'], out_meas['image']), "image mismatch"
    assert np.array_equal(out_render['visible_mask'], out_meas['visible_mask']), "visible_mask mismatch"
    assert np.allclose(out_render['depth'], out_meas['depth']), "depth mismatch"


def test_batched_measurement_interface():
    """
    Verify that BatchedPinholeCamera.measurement() returns the same output as render().
    """
    torch = pytest.importorskip("torch")
    from rotorpy.sensors.camera import BatchedPinholeCamera
    from rotorpy.world import World

    world = World.empty((-2, 4, -2, 4, -1, 5), add_features=True,
                        feature_mode='regular', feature_spacing=0.5, descriptor_noise=0.0)
    bcam = BatchedPinholeCamera(num_drones=2)

    states = {'x': torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
              'q': torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])}

    out_render = bcam.render(world, states)
    out_meas = bcam.measurement(world, states)

    assert out_render.keys() == out_meas.keys(), "measurement() and render() should return the same keys"
    assert torch.allclose(out_render['image'], out_meas['image']), "image mismatch"
    assert torch.equal(out_render['visible_mask'], out_meas['visible_mask']), "visible_mask mismatch"
    assert torch.allclose(out_render['depth'], out_meas['depth']), "depth mismatch"

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

"""
Functions for visualizing the output of a pinhole camera sensor.

"""

def draw_camera_frustum(ax, camera_pose, intrinsics, scale=1.0, color='k', alpha=1.0):
    """
    Draw a wireframe frustum on a 3D axis at a camera pose.

    The camera looks along +z. The 4 image-corner ray directions are computed
    from the intrinsics, scaled by `scale` meters, and transformed from the
    camera frame to the world frame using the pose quaternion.

    Inputs:
        ax, Axes3D object
        camera_pose, dict with keys
            x, camera world position, shape=(3,)
            q, world-to-camera quaternion [i, j, k, w], shape=(4,)
        intrinsics, dict with keys fx, fy, cx, cy, width, height
        scale, length in meters of the frustum (distance from the apex to the
            image-plane rectangle), default is 1.0
        color, color of the frustum lines, default is 'r'
        alpha, transparency of the frustum lines, default is 0.5

    Outputs:
        artists, list of the Line artists that were added to the axis
    """
    fx = float(intrinsics['fx'])
    fy = float(intrinsics['fy'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])
    width = float(intrinsics['width'])
    height = float(intrinsics['height'])

    # Ray directions of the 4 image corners in the camera frame (x right,
    # y down, z forward). Normalize and scale to the requested frustum length.
    corners_uv = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    dirs_cam = np.stack([np.array([(u - cx) / fx, (v - cy) / fy, 1.0]) for u, v in corners_uv], axis=0)
    dirs_cam = (dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)) * scale

    # Transform the ray directions from the camera frame into the world frame.
    x = np.asarray(camera_pose['x'], dtype=np.float64)
    q = np.asarray(camera_pose['q'], dtype=np.float64)
    R_WC = Rotation.from_quat(q).as_matrix()  # world -> camera
    R_CW = R_WC.T  # camera -> world
    corners_world = x + dirs_cam @ R_CW.T  # (4, 3)

    # Apex at the camera position, with lines to each corner and a rectangle
    # connecting the corners.
    artists = []
    for corner in corners_world:
        artists.append(ax.plot([x[0], corner[0]], [x[1], corner[1]], [x[2], corner[2]],
                               color=color, alpha=alpha))
    for i in range(4):
        a = corners_world[i]
        b = corners_world[(i + 1) % 4]
        artists.append(ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                               color=color, alpha=alpha))

    return [artist for line in artists for artist in line]

def draw_camera_triad(ax, camera_pose, scale=1.0, alpha=1.0):
    """
    Draw a coordinate triad marking the camera frame axes at the camera origin.

    The camera frame is right-handed with x right, y down, and z forward. The
    triad lines are colored red (x), green (y), and blue (z) in the camera
    frame, transformed into the world frame using the pose quaternion.

    Inputs:
        ax, Axes3D object
        camera_pose, dict with keys
            x, camera world position, shape=(3,)
            q, world-to-camera quaternion [i, j, k, w], shape=(4,)
        scale, length in meters of each triad arm, default is 1.0
        alpha, transparency of the triad lines, default is 1.0

    Outputs:
        artists, list of the Line artists that were added to the axis
    """
    x = np.asarray(camera_pose['x'], dtype=np.float64)
    q = np.asarray(camera_pose['q'], dtype=np.float64)
    R_WC = Rotation.from_quat(q).as_matrix()  # world -> camera
    R_CW = R_WC.T  # camera -> world

    artists = []
    for i, color in enumerate(['r', 'g', 'b']):
        axis_cam = np.zeros(3)
        axis_cam[i] = scale
        end_world = x + axis_cam @ R_CW.T
        artists.append(ax.plot([x[0], end_world[0]], [x[1], end_world[1]], [x[2], end_world[2]],
                               color=color, alpha=alpha))

    return [artist for line in artists for artist in line]

def plot_world_with_camera(ax, world, camera_pose=None, show_features=True, alpha=0.7,
                           frustum_scale=None, intrinsics=None, show_drone=False,
                           drone_state=None, drone_scale_factor=1.0):
    """
    Plot a 3D scene of a world with its surface features and an optional camera
    frustum, modifying the axis in place.

    Inputs:
        ax, Axes3D object
        world, World object exposing draw(), get_surface_features(),
            get_feature_descriptors(), and world['bounds']['extents']
        camera_pose, dict with keys 'x' and 'q' describing the camera pose in
            the world frame, or None to skip the frustum, default is None
        show_features, if True and the world has surface features, scatter them
            colored by their descriptors, default is True
        alpha, transparency of the world blocks, default is 0.7
        frustum_scale, length in meters of the camera frustum, or None to
            auto-scale to the world size, default is None
        intrinsics, dict with keys fx, fy, cx, cy, width, height used to size
            the camera frustum, or None to skip the frustum, default is None
        show_drone, if True render the drone in the 3D scene, default is False
        drone_state, dict with keys 'x' (3,) and 'q' (4,) describing the
            drone pose, required when show_drone is True, default is None
        drone_scale_factor, scale factor for the wind arrow, default is 1.0

    Outputs:
        None (modifies the axis in place)
    """
    world.draw(ax, alpha=alpha)

    if show_features:
        features = world.get_surface_features()
        if features is not None and len(features) > 0:
            descriptors = world.get_feature_descriptors()
            if descriptors is None:
                c = np.full((len(features), 3), 0.6)
            else:
                c = descriptors
            ax.scatter(features[:, 0], features[:, 1], features[:, 2],
                       s=6, c=c, edgecolors='none', depthshade=False)

    if camera_pose is not None and intrinsics is not None:
        if frustum_scale is None:
            extents = world.world['bounds']['extents']
            frustum_scale = 0.1 * np.max(np.array(extents)[1::2] - np.array(extents)[0::2])
        draw_camera_frustum(ax, camera_pose, intrinsics, scale=frustum_scale)
        draw_camera_triad(ax, camera_pose, scale=0.50*frustum_scale)

    if show_drone and drone_state is not None:
        from rotorpy.utils.shapes import Quadrotor
        drone_quad = Quadrotor(ax, wind=False, shade=True)
        R_drone = Rotation.from_quat(drone_state['q']).as_matrix()
        drone_quad.transform(
            position=np.array(drone_state['x'], dtype=float).copy(),
            rotation=R_drone)

    (xmin, xmax, ymin, ymax, zmin, zmax) = world.world['bounds']['extents']
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_zlim((zmin, zmax))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return

def plot_camera_view(camera, world, state, ax3d=None, ax_img=None, save_path=None,
                     frustum_scale=1.0, show_keypoints=False, render_kwargs=None,
                     show_drone=False):
    """
    Plot a figure with the 3D scene of a world (including a camera frustum) on
    the left and the rendered camera image on the right.

    Inputs:
        camera, PinholeCamera object exposing compute_camera_pose(), render(),
            and intrinsics
        world, World object exposing draw(), get_surface_features(),
            get_feature_descriptors(), and world['bounds']['extents']
        state, dict describing the vehicle state with keys 'x' (3,) and
            'q' (4,) [i, j, k, w]
        ax3d, Axes3D object to use for the scene, or None to create one,
            default is None
        ax_img, matplotlib axis to use for the image, or None to create one,
            default is None
        save_path, filename to save the figure to, or None to skip saving,
            default is None
        frustum_scale, length in meters of the camera frustum, default is 1.0
        show_keypoints, if True overlay '+' markers on the visible keypoints,
            default is False
        render_kwargs, dict of additional keyword arguments forwarded to
            camera.render() (e.g. splat_radius), default is None
        show_drone, if True render the drone in the 3D scene, default is False

    Outputs:
        (fig, (ax3d, ax_img)) tuple with the figure and the two axes
    """
    if ax3d is None or ax_img is None:
        fig = plt.figure(figsize=(11.0, 5.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
        ax3d = fig.add_subplot(gs[0], projection='3d')
        ax_img = fig.add_subplot(gs[1])
    else:
        fig = ax3d.get_figure()

    camera_pose = camera.compute_camera_pose(state)
    plot_world_with_camera(ax3d, world, camera_pose=camera_pose,
                           intrinsics=camera.intrinsics, frustum_scale=frustum_scale,
                           show_drone=show_drone, drone_state=state)
    ax3d.set_aspect('equal')

    if render_kwargs is None:
        render_kwargs = {}
    render = camera.render(world, state, **render_kwargs)
    # 'nearest' keeps single-feature splats from being antialiased away.
    ax_img.imshow(render['image'], origin='upper', interpolation='nearest')
    if show_keypoints:
        keypoints = render['keypoints']
        if keypoints is not None and len(keypoints) > 0:
            ax_img.plot(keypoints[:, 0], keypoints[:, 1], '+', color='k', markersize=4)

    ax3d.set_title('3D Scene')
    ax_img.set_title('Camera View')

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return (fig, (ax3d, ax_img))

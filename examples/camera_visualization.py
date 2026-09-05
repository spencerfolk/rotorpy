"""
Camera visualization example for RotorPy.

One large figure showcasing the pinhole camera sensor:

  Left pane   - the 3D world (two cuboids) with the camera pose triad.
  Right pane  - a 4x4 grid of rendered frames sweeping two settings:
                  rows:     lens intrinsics, normal (top) -> wide angle (bottom),
                            i.e. decreasing focal length with increasing barrel
                            distortion.
                  columns:  measurement() visual noise effect, feature_rate
                            0 (left, off) -> 250 (right, heavy).

Each grid cell is labeled top-left with its focal length, radial distortion
coefficient k1, and noise feature_rate. A fixed noise seed is used so the
injected feature positions are identical across lenses, isolating the effect of
the lens from the effect of the noise.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rotorpy.world import World
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.utils.camera_plotter import draw_camera_triad


def main():
    # World: two axis-aligned cuboids. Cuboid A sits in front of (closer to the
    # camera than) cuboid B, so A partially occludes B from the camera.
    world_data = {
        'bounds': {'extents': [0.0, 6.0, 0.0, 4.0, 0.0, 3.0]},
        'blocks': [
            # Near cuboid: small, red, in the middle of the frame.
            {'extents': [2.2, 2.6, 1.5, 1.9, 0.9, 1.3], 'color': [0.9, 0.1, 0.1]},
            # Far cuboid: large, blue, extends beyond A on every side so its rim
            # peeks out all around the red cuboid in the rendered image.
            {'extents': [3.6, 5.6, 0.7, 2.5, 0.3, 2.1], 'color': [0.1, 0.3, 0.9]},
        ],
    }
    # add_features=True places visual features (3D position + color) on the
    # surfaces of the world geometry. descriptor_noise jitters the feature
    # colors so each feature looks distinct.
    world = World(world_data, add_features=True, feature_mode='random',
                  N_features_per_surface=400, descriptor_noise=0.1)

    # Camera extrinsics: position and orientation of the camera relative to the
    # vehicle body. The orientation is a quaternion [i, j, k, w] describing the
    # rotation from the body frame to the camera frame. With the vehicle parked
    # at identity attitude this is also the camera pose in the world frame.
    extrinsics = {'position': np.array([0.0, 0.0, 0.0]),
                  'orientation': (Rotation.from_euler('x', 25, degrees=True) *
                                  Rotation.from_euler('y', -90, degrees=True) *
                                  Rotation.from_euler('x', 90, degrees=True)).as_quat()}

    # Park the vehicle (identity orientation) to the -x side of both cuboids,
    # centered on their common midline, and look along +x. The camera therefore
    # sees the red cuboid in front and the blue cuboid peeking out around it.
    state = {'x': np.array([1.0, 1.6, 2.4]),
             'q': np.array([0.0, 0.0, 0.0, 1.0])}

    # Shared image geometry; only the focal length and distortion vary per row.
    base_intrinsics = {'width': 480, 'height': 360, 'cx': 240.0, 'cy': 180.0}

    # Rows: lens intrinsics, top (normal lens) to bottom (wide angle). Smaller
    # focal length -> wider field of view, paired with stronger barrel (k1<0)
    # radial distortion.
    lens_rows = [
        (600.0, [0.00, 0.00, 0.0, 0.0, 0.0]),
        (480.0, [-0.20, 0.05, 0.0, 0.0, 0.0]),
        (360.0, [-0.40, 0.15, 0.0, 0.0, 0.0]),
        (260.0, [-0.60, 0.30, 0.0, 0.0, 0.0]),
    ]
    # Columns: measurement() visual noise, left (off) to right (heavy). The
    # per-frame count is Poisson(feature_rate).
    noise_levels = np.linspace(0, 250, 4, dtype=int)
    rows, cols = len(lens_rows), len(noise_levels)

    # Render one frame per (lens, noise) combination. measurement() applies the
    # noise effect on top of the raw render(). A fixed seed keeps injected
    # feature positions identical across lenses so only the lens differs.
    images = []
    for fx, dist_coeffs in lens_rows:
        row_images = []
        for rate in noise_levels:
            intrinsics = dict(base_intrinsics)
            intrinsics.update({'fx': fx, 'fy': fx, 'dist_coeffs': np.array(dist_coeffs)})
            noise_params = None if rate == 0 else {'feature_rate': rate,
                                                   'splat_radius': 2,
                                                   'intensity': 0.9,
                                                   'seed': 0}
            camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics,
                                   splat_radius=3, noise_params=noise_params)
            row_images.append(camera.measurement(state, world)['image'])
        images.append(row_images)

    # ------------------------------------------------------------------
    # Figure: world + camera pose (left), 4x4 render grid (right).
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.6], wspace=0.15,
                          left=0.03, right=0.98, top=0.95, bottom=0.05)
    grid = gs[0, 1].subgridspec(rows + 1, cols + 1,
                                width_ratios=[0.08] + [1.0] * cols,
                                height_ratios=[0.07] + [1.0] * rows,
                                wspace=0.12, hspace=0.20)

    # Left pane: the world and the camera pose triad (only the triad - no
    # frustum - keeps the map clean).
    ax_map = fig.add_subplot(gs[0, 0], projection='3d')
    world.draw(ax_map)
    pose = camera.compute_camera_pose(state)
    draw_camera_triad(ax_map, pose, scale=0.7)
    ax_map.set_title('World + camera pose', fontsize=14)

    # Grid column header: noise increases left -> right.
    ax_noise = fig.add_subplot(grid[0, 1:])
    ax_noise.axis('off')
    ax_noise.text(0.5, 0.5, 'Noise Level', ha='center', va='center',
                  fontsize=17, fontweight='bold')

    # Grid row header: FOV widens top -> bottom.
    ax_fov = fig.add_subplot(grid[1:, 0])
    ax_fov.axis('off')
    ax_fov.text(0.5, 0.5, 'FOV', ha='center', va='center', rotation=90,
                fontsize=17, fontweight='bold')

    # The 4x4 grid of rendered frames.
    for i in range(rows):
        for j in range(cols):
            ax = fig.add_subplot(grid[i + 1, j + 1])
            ax.imshow(images[i][j], origin='upper', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            fx, dist_coeffs = lens_rows[i]
            label = 'f=%d  k1=%+.2f\nnoise=%d' % (fx, dist_coeffs[0], noise_levels[j])
            ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=8, va='top',
                    color='k', bbox=dict(boxstyle='round,pad=0.2', fc='w',
                                         alpha=0.75, ec='none'))

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'camera_visualization.png')
    fig.savefig(save_path, dpi=120)
    print('Figure saved to:', os.path.abspath(save_path))
    plt.show()


if __name__ == "__main__":
    main()
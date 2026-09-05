"""
Camera visualization example for RotorPy.

Two figures showcase the pinhole camera sensor and the world feature model.

Figure 1 (camera_visualization.png) - the lens / noise sweep:
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

Figure 2 (camera_feature_modes.png) - the feature-type sweep:
    Left pane   - the same 3D world with its surface features and camera triad.
    Right pane  - a 4x4 grid of rendered frames from a single nominal camera
                  (no lens distortion, no noise), sweeping how features are
                  placed on the world geometry:
                    rows:     the four feature types
                              regular grid, random splatter, uniform edges,
                              and random edges.
                    columns:  feature density, sparse (left) -> dense (right).
    Each cell is a fresh world built from one generator configuration carrying
    128-d synthetic descriptor vectors, so the grid contrasts how each feature
    type covers the surfaces (and image) at increasing density. Each cell is
    labeled with its spacing/density setting and total feature count.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rotorpy.world import World
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.utils.camera_plotter import draw_camera_triad, feature_colors


# World: two axis-aligned cuboids. Cuboid A sits in front of (closer to the
# camera than) cuboid B, so A partially occludes B from the camera.
WORLD_DATA = {
    'bounds': {'extents': [0.0, 6.0, 0.0, 4.0, 0.0, 3.0]},
    'blocks': [
        # Near cuboid: small, red, in the middle of the frame.
        {'extents': [2.2, 2.6, 1.5, 1.9, 0.9, 1.3], 'color': [0.9, 0.1, 0.1]},
        # Far cuboid: large, blue, extends beyond A on every side so its rim
        # peeks out all around the red cuboid in the rendered image.
        {'extents': [3.6, 5.6, 0.7, 2.5, 0.3, 2.1], 'color': [0.1, 0.3, 0.9]},
    ],
}

# Camera extrinsics: position and orientation of the camera relative to the
# vehicle body. The orientation is a quaternion [i, j, k, w] describing the
# rotation from the body frame to the camera frame. With the vehicle parked at
# identity attitude this is also the camera pose in the world frame.
EXTRINSICS = {'position': np.array([0.0, 0.0, 0.0]),
              'orientation': (Rotation.from_euler('x', 25, degrees=True) *
                              Rotation.from_euler('y', -90, degrees=True) *
                              Rotation.from_euler('x', 90, degrees=True)).as_quat()}

# Park the vehicle (identity orientation) to the -x side of both cuboids,
# centered on their common midline, and look along +x. The camera therefore
# sees the red cuboid in front and the blue cuboid peeking out around it.
STATE = {'x': np.array([1.0, 1.6, 2.4]),
         'q': np.array([0.0, 0.0, 0.0, 1.0])}

# Shared image geometry; only the focal length and distortion vary per row.
BASE_INTRINSICS = {'width': 480, 'height': 360, 'cx': 240.0, 'cy': 180.0}


def _world(mode, **generator_kwargs):
    """
    Build the demo world with surface features from a specific generator.
    Every world carries jittered colors and 128-d synthetic descriptor
    vectors (e.g. SIFT-style), the full per-feature representation.
    """
    return World(WORLD_DATA, add_features=True, feature_mode=mode,
                 descriptor_noise=0.15, descriptor_dim=128, **generator_kwargs)


def _draw_scene(ax, world, pose):
    """
    Draw the world geometry, its surface features, and the camera pose triad.
    """
    world.draw(ax)
    features = world.get_surface_features()
    if features is not None and features.shape[0] > 0:
        colors = feature_colors(world, features.shape[0])
        ax.scatter(features[:, 0], features[:, 1], features[:, 2],
                   s=6, c=colors, depthshade=False, edgecolors='none')
    draw_camera_triad(ax, pose, scale=0.7)
    ax.set_title('World + camera pose', fontsize=14)


def _new_grid(rows, cols):
    """
    Figure with the world pane on the left and a rows x cols render grid on the
    right, matching the layout of both visualization figures.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.6], wspace=0.15,
                          left=0.03, right=0.98, top=0.95, bottom=0.05)
    grid = gs[0, 1].subgridspec(rows + 1, cols + 1,
                                width_ratios=[0.10] + [1.0] * cols,
                                height_ratios=[0.07] + [1.0] * rows,
                                wspace=0.12, hspace=0.20)
    return fig, gs, grid


def _add_cell_label(ax, text):
    """
    Draw a small white-label box in the top-left corner of a rendered frame.
    """
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=8, va='top',
            color='k', bbox=dict(boxstyle='round,pad=0.2', fc='w',
                                 alpha=0.75, ec='none'))


def figure_lens_noise():
    """
    Figure 1: sweep lens intrinsics (rows) and measurement() noise (columns).
    """
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

    # Reference world: random splatter at a uniform areal density with 96-d
    # ALIKED-style descriptor vectors.
    world = World(WORLD_DATA, add_features=True, feature_mode='random',
                  feature_density=400, descriptor_noise=0.1,
                  descriptor_dim=96)
    camera = PinholeCamera(intrinsics=dict(BASE_INTRINSICS,
                                           fx=600.0, fy=600.0,
                                           dist_coeffs=np.zeros(5)),
                           extrinsics=EXTRINSICS, splat_radius=3)

    # Render one frame per (lens, noise) combination. measurement() applies the
    # noise effect on top of the raw render(). A fixed seed keeps injected
    # feature positions identical across lenses so only the lens differs.
    images = []
    for fx, dist_coeffs in lens_rows:
        row_images = []
        for rate in noise_levels:
            intrinsics = dict(BASE_INTRINSICS)
            intrinsics.update({'fx': fx, 'fy': fx, 'dist_coeffs': np.array(dist_coeffs)})
            noise_params = None if rate == 0 else {'feature_rate': rate,
                                                   'splat_radius': 2,
                                                   'intensity': 0.9,
                                                   'seed': 0}
            cam = PinholeCamera(intrinsics=intrinsics, extrinsics=EXTRINSICS,
                                splat_radius=3, noise_params=noise_params)
            row_images.append(cam.measurement(STATE, world)['image'])
        images.append(row_images)

    fig, gs, grid = _new_grid(rows, cols)

    # Left pane: the world and the camera pose triad (only the triad - no
    # frustum - keeps the map clean).
    _draw_scene(fig.add_subplot(gs[0, 0], projection='3d'), world,
                camera.compute_camera_pose(STATE))

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
            _add_cell_label(ax, 'f=%d  k1=%+.2f\nnoise=%d' % (fx, dist_coeffs[0],
                                                              noise_levels[j]))

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'camera_visualization.png')
    fig.savefig(save_path, dpi=120)
    print('Camera lens/noise figure saved to:', os.path.abspath(save_path))


def figure_feature_modes():
    """
    Figure 2: one nominal camera, no noise, sweeping the world feature
    generators row-by-row and feature density column-by-column.
    """
    # (mode, generator kwarg, parameter label, unit, column values).
    # Columns go sparse (left) -> dense (right) for each feature type. The ball
    # park differs per type, so every row has its own per-column parameter.
    modes = [
        ('regular',      'feature_spacing', 'spacing',      'm',       [0.25, 0.15, 0.10, 0.06]),
        ('random',       'feature_density', 'density',      'per m^2', [50, 100, 180, 400]),
        ('edge_uniform', 'edge_spacing',    'edge spacing', 'm',       [0.8, 0.4, 0.1, 0.05]),
        ('edge_random',  'edge_density',    'edge density', 'per m',   [2, 4, 8, 16]),
    ]
    mode_names = ['regular grid', 'random splatter', 'uniform edges', 'random edges']
    rows, cols = len(modes), 4

    # Nominal camera: narrow lens (no distortion) + splatted features, no noise.
    intrinsics = dict(BASE_INTRINSICS, fx=480.0, fy=480.0, dist_coeffs=[-0.20, 0.05, 0.0, 0.0, 0.0])
    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=EXTRINSICS,
                           splat_radius=3)

    # Reference world for the left pane: random splatter at middling density.
    ref_world = _world('random', feature_density=180)

    fig, gs, grid = _new_grid(rows, cols)

    # Left pane: world geometry + features + camera triad.
    _draw_scene(fig.add_subplot(gs[0, 0], projection='3d'), ref_world,
                camera.compute_camera_pose(STATE))

    # Grid column header: feature density increases left -> right.
    ax_density = fig.add_subplot(grid[0, 1:])
    ax_density.axis('off')
    ax_density.text(0.5, 0.5, 'Feature Density', ha='center', va='center',
                    fontsize=17, fontweight='bold')

    # Grid row headers: the four feature types.
    for r, name in enumerate(mode_names):
        ax_mode = fig.add_subplot(grid[r + 1, 0])
        ax_mode.axis('off')
        ax_mode.text(0.5, 0.5, name, ha='center', va='center', rotation=90,
                     fontsize=15, fontweight='bold')

    # The 4x4 grid of rendered frames, one fresh world per cell.
    for r, (mode, key, param_name, unit, levels) in enumerate(modes):
        for c, value in enumerate(levels):
            world = _world(mode, **{key: value})
            out = camera.render(world, STATE)
            n_feats = world.get_surface_features().shape[0]

            ax = fig.add_subplot(grid[r + 1, c + 1])
            ax.imshow(out['image'], origin='upper', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            _add_cell_label(ax, '%s %g %s\n%d feats' % (param_name, value, unit, n_feats))

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'camera_feature_modes.png')
    fig.savefig(save_path, dpi=120)
    print('Camera feature-mode figure saved to:', os.path.abspath(save_path))


def main():
    figure_lens_noise()
    figure_feature_modes()
    plt.show()


if __name__ == "__main__":
    main()
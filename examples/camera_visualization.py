"""
Camera visualization example for RotorPy.

A simple, self-contained demonstration of the pinhole camera sensor on a custom world with just two cuboids with different colors and sizes.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rotorpy.world import World
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.utils.camera_plotter import plot_camera_view


def main():
    # Custom world: two axis-aligned cuboids. Cuboid A sits in front of (closer
    # to the camera than) cuboid B, so A partially occludes B from the camera.
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
    # add_features=True places visual features (3D position + color) on the surfaces of the world geometry.
    #   feature_mode='regular' places features in a regular grid pattern with fixed feature spacing. 
    #   feature_mode='random' places N_features_per_surface random features on each surface. 
    #   The descriptor_noise param adds Gaussian noise to the feature descriptors (colors) to simulate real-world sensor noise or unique feature descriptors.
    world = World(world_data, add_features=True, feature_mode='regular',
                  feature_spacing=0.05, descriptor_noise=0.1)
    # world = World(world_data, add_features=True, feature_mode='random',
    #                N_features_per_surface=100, descriptor_noise=0.1)

    # Camera intrinsics: focal lengths, image size, principal point, and
    # distortion coefficients [k1, k2, p1, p2, k3] (zeros => pinhole only).
    intrinsics = {'fx': 600.0, 'fy': 600.0, 'width': 640, 'height': 480,
                  'cx': 320.0, 'cy': 240.0,
                  'dist_coeffs': [-0.3, 0.1, 0.0, 0.0, 0.0]}
    # Camera extrinsics: position and orientation of the camera relative to the
    # vehicle body. The orientation is a quaternion [i, j, k, w] describing the
    # rotation from the body frame to the camera frame.
    extrinsics = {'position': np.array([0.0, 0.0, 0.0]),
                  'orientation': (Rotation.from_euler('x', 25, degrees=True)*Rotation.from_euler('y', -90, degrees=True)*Rotation.from_euler('x', 90, degrees=True)).as_quat()}
    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)

    # Park the vehicle (identity orientation) to the -x side of both cuboids,
    # centered on their common midline, and look along +x. The camera therefore
    # sees the red cuboid in front and the blue cuboid peeking out around it.
    state = {'x': np.array([1.0, 1.6, 2.4]),
             'q': np.array([0.0, 0.0, 0.0, 1.0])}

    # Render the world from the camera. The render output includes the image,
    # the visible feature keypoints, and per-feature depths/visibility masks.
    render = camera.render(world, state)
    print("Render stats:")
    print("  total features:", render['visible_mask'].size)
    print("  visible keypoints:", render['keypoints'].shape[0])
    print("  image shape:", render['image'].shape)

    # Produce the side-by-side figure (3D scene + camera view) and save it.
    media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(media_dir, exist_ok=True)
    save_path = os.path.join(media_dir, 'camera_visualization.png')
    # splat_radius controls the pixel size of each feature splat; a larger
    # radius makes the features read as near-solid colored regions.
    fig, _ = plot_camera_view(camera, world, state, show_drone=True, save_path=save_path,
                              render_kwargs={'splat_radius': 4})
    import matplotlib.pyplot as plt
    plt.show()
    plt.close(fig)
    print("Figure saved to:", os.path.abspath(save_path))


if __name__ == "__main__":
    main()

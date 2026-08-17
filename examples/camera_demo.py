"""
Camera circling example.

Runs a simulation of a drone circling the double-pillar world (same trajectory
as basic_usage.py), then produces an animation showing the 3D scene with a
camera frustum alongside the rendered camera view.

At each timestep the camera is placed at the drone's position with roll and
pitch taken from the drone's attitude but the yaw is replaced so the camera
always points inward toward the origin (center of the circle).
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.world import World
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.utils.camera_plotter import plot_camera_view


def main():
    # ------------------------------------------------------------------
    # 1. Load world and run the simulation (same setup as basic_usage.py)
    # ------------------------------------------------------------------
    world = World.from_file(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'rotorpy',
        'worlds', 'double_pillar.json'))
    world.generate_surface_features(mode='regular', spacing=0.05, descriptor_noise=0.1)

    sim_instance = Environment(
        vehicle=Multirotor(quad_params),
        controller=SE3Control(quad_params),
        trajectory=ThreeDCircularTraj(
            radius=np.array([3, 3, 0]),
            freq=np.array([0.15, 0.15, 0])),
        world=world,
        sim_rate=100,
        safety_margin=0.25,
    )

    x0 = {'x': np.array([0, 0, 0]),
           'v': np.zeros(3),
           'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
           'w': np.zeros(3),
           'wind': np.array([0, 0, 0]),
           'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
    sim_instance.vehicle.initial_state = x0

    print("Running simulation...")
    results = sim_instance.run(
        t_final=20,
        plot=False,
        animate_bool=False,
        verbose=True,
    )

    time_hist = results['time']       # (N,)
    x_hist = results['state']['x']    # (N, 3)
    q_hist = results['state']['q']    # (N, 4)  [i, j, k, w]
    N = len(time_hist)
    print(f"Simulation produced {N} timesteps.")

    # ------------------------------------------------------------------
    # 2. Set up the camera
    # ------------------------------------------------------------------
    # Forward-facing camera: camera +z (optical axis) aligned with body +x.
    # No pitch tilt relative to the body frame.
    base_orientation = Rotation.from_euler('y', -90, degrees=True)*Rotation.from_euler('x', 90, degrees=True)
    extrinsics = {
        'position': np.array([0.0, 0.0, 0.0]),
        'orientation': base_orientation.as_quat(),
    }

    intrinsics = {
        'fx': 600.0, 'fy': 600.0,
        'width': 640, 'height': 480,
        'cx': 320.0, 'cy': 240.0,
        'dist_coeffs': [-0.3, 0.1, 0.0, 0.0, 0.0], # wide lens
    }

    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)

    # ------------------------------------------------------------------
    # 3. Subsample the trajectory for animation
    # ------------------------------------------------------------------
    step = 10  # pick every 10th sim step (10 Hz render vs 100 Hz sim)
    indices = list(range(0, N, step))

    # Pre-compute the inward-pointing camera states.
    cam_states = []
    for i in indices:
        pos = x_hist[i]
        q_drone = q_hist[i]

        # Decompose the drone quaternion into yaw, pitch, roll (ZYX order).
        rpy = Rotation.from_quat(q_drone).as_euler('ZYX')  # [yaw, pitch, roll]
        pitch = rpy[1]
        roll = rpy[2]

        # Inward-pointing yaw: from the drone toward the origin in the xy plane.
        inward_yaw = np.arctan2(-pos[1], -pos[0])

        # Reconstruct quaternion with the new yaw.
        R_drone = Rotation.from_euler('ZYX', [inward_yaw, pitch, roll])

        cam_states.append({'x': pos.copy(), 'q': (R_drone.inv()).as_quat()})

    # ------------------------------------------------------------------
    # 4. Build the animation
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax_img = fig.add_subplot(gs[1])

    frustum_scale = 1.0

    def _update(frame_idx):
        ax3d.clear()
        ax_img.clear()

        state = cam_states[frame_idx]
        plot_camera_view(camera, world, state,
                     ax3d=ax3d, ax_img=ax_img,
                     frustum_scale=frustum_scale,
                     render_kwargs={'splat_radius': 3})

        t = time_hist[indices[frame_idx]]
        fig.suptitle(f't = {t:.2f} s', fontsize=12)
        return []

    print(f"Rendering animation ({len(indices)} frames)...")
    anim = animation.FuncAnimation(
        fig, _update, frames=len(indices),
        interval=50, blit=False)

    # Save as gif.
    media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(media_dir, exist_ok=True)
    save_path = os.path.join(media_dir, 'camera_demo.gif')
    anim.save(save_path, writer='pillow', fps=20)
    print(f"Animation saved to {os.path.abspath(save_path)}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

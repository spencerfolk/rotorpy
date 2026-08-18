"""
Camera circling example.

Runs a simulation of a drone circling the double-pillar world (same trajectory
as basic_usage.py), then produces an animation showing the 3D scene with a
camera frustum alongside the rendered camera view.

The drone's yaw is commanded by a custom yaw trajectory (InwardYawFromTraj) that
always points the camera inward toward the center of the circle.
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


class InwardYawFromTraj:
    """
    Yaw trajectory that points inward toward a circle center.

    This version takes the circle center and computes yaw from the drone's
    position at each timestep. Since update() receives x_dot but not x,
    we store the position externally (updated by the trajectory) or compute
    the inward yaw analytically for a known circular trajectory.
    """
    def __init__(self, center, radius, freq):
        """
        Parameters:
            center, (3,) center of the circle
            radius, (3,) radius of the circle (only x,y used for inward yaw)
            freq, (3,) frequency of the circle
        """
        self.center = np.array(center, dtype=float)
        self.radius = np.array(radius, dtype=float)
        self.omega = 2 * np.pi * np.array(freq, dtype=float)
        self._prev_yaw = 0.0
        self._prev_t = None
        self._prev_yaw_raw = 0.0

    def update(self, t, x_dot=None):
        """
        Compute yaw pointing inward toward center.

        For a circle parameterized as:
            x = cx + Rx*cos(wx*t)
            y = cy + Ry*sin(wy*t)
        The inward vector from the drone toward the center is (-cos(wx*t), -sin(wy*t)),
        so yaw = atan2(-sin(wy*t), -cos(wx*t)).
        """
        # Inward direction from drone position toward center
        yaw = np.arctan2(-np.sin(self.omega[1]*t), -np.cos(self.omega[0]*t))

        # Finite-difference yaw rate with wraparound handling
        if self._prev_t is not None and t > self._prev_t:
            dt = t - self._prev_t
            dyaw = yaw - self._prev_yaw_raw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            yaw_dot = dyaw / dt
        else:
            yaw_dot = 0.0

        yaw_ddot = 0.0
        self._prev_t = t
        self._prev_yaw_raw = yaw

        return {'yaw': yaw, 'yaw_dot': yaw_dot, 'yaw_ddot': yaw_ddot}


def main():
    # ------------------------------------------------------------------
    # 1. Load world and run the simulation
    # ------------------------------------------------------------------
    world = World.from_file(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'rotorpy',
        'worlds', 'double_pillar.json'))
    world.generate_surface_features(mode='regular', spacing=0.05, descriptor_noise=0.1)

    circle_center = np.array([0, 0, 0])
    circle_radius = np.array([3, 3, 0])
    circle_freq = np.array([0.2, 0.2, 0])

    sim_instance = Environment(
        vehicle=Multirotor(quad_params),
        controller=SE3Control(quad_params),
        trajectory=ThreeDCircularTraj(
            center=circle_center,
            radius=circle_radius,
            freq=circle_freq,
            yaw_traj=InwardYawFromTraj(circle_center, circle_radius, circle_freq)),
        world=world,
        sim_rate=100,
        safety_margin=0.25,
    )

    yaw_init = InwardYawFromTraj(circle_center, circle_radius, circle_freq).update(0.0)['yaw']
    q_init = Rotation.from_euler('z', yaw_init).as_quat()  # [i,j,k,w]

    x0 = {'x': np.array([0, 0, 0]),
           'v': np.zeros(3),
           'q': q_init,  # [i,j,k,w]
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
    # Forward-facing camera: camera +z (optical axis) aligned with body +x, with slight upward tilt.
    base_orientation = Rotation.from_euler('x', -45, degrees=True)*Rotation.from_euler('y', -90, degrees=True)*Rotation.from_euler('x', 90, degrees=True)
    extrinsics = {
        'position': np.array([0.0, 0.0, 0.0]),
        'orientation': base_orientation.as_quat(),
    }

    intrinsics = {
        'fx': 400.0, 'fy': 400.0,
        'width': 640, 'height': 480,
        'cx': 320.0, 'cy': 240.0,
        'dist_coeffs': [-0.3, 0.1, 0.0, 0.0, 0.0], # wide lens
    }

    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)

    # ------------------------------------------------------------------
    # 3. Subsample the trajectory for animation
    # ------------------------------------------------------------------
    step = 4  # pick every 4th sim step (25 Hz render vs 100 Hz sim)
    indices = list(range(0, N, step))

    # Pre-compute the inward-pointing camera states.
    cam_states = []
    for i in indices:
        pos = x_hist[i]
        q_drone = q_hist[i]
        cam_states.append({'x': pos.copy(), 'q': q_drone})

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
                     show_drone=True,
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
    anim.save(save_path, writer='pillow', fps=25)
    print(f"Animation saved to {os.path.abspath(save_path)}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

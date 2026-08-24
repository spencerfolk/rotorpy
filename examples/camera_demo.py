"""
Primary camera demo.

Runs a simulation of a UAV circling the double-pillar world (same trajectory
as basic_usage.py) with an onboard PinholeCamera. The camera is passed to the
Environment, which captures frames during the run at the camera's frame_rate.
Frames are shown in the plots/animation alongside the 3D scene, and can be
saved to a .npz archive (plus optional pngs) via save_camera_data(); they are
deliberately excluded from save_to_csv() since images are far too large for a
csv.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.world import World
from rotorpy.wind.dryden_winds import DrydenGust
from rotorpy.sensors.camera import PinholeCamera

# Custom yaw trajectory to point the camera inward toward the circle center.
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
    # 1. Load world and set up the onboard camera
    # ------------------------------------------------------------------
    world = World.from_file(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'rotorpy',
        'worlds', 'double_pillar.json'))
    # Generate surface features for the world. This adds visual features (3D position + color) on the surfaces of the world geometry.
    world.generate_surface_features(mode='regular', spacing=0.08, descriptor_noise=0.15)  # regular grid feature generation
    # world.generate_surface_features(mode='random', N_features_per_surface=200, descriptor_noise=0.15) # random feature generation

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

    # frame_rate decouples the capture rate from sim_rate; None would render
    # every simulation step. splat_radius enlarges each feature's pixel patch
    # so features are easier to see in the rendered frames.
    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics,
                           frame_rate=25, splat_radius=3)

    circle_center = np.array([0, 0, 0])
    circle_radius = np.array([3, 3, 0])
    circle_freq = np.array([0.2, 0.2, 0])

    # ------------------------------------------------------------------
    # 2. Construct the environment, passing in the camera
    # ------------------------------------------------------------------
    sim_instance = Environment(
        vehicle=Multirotor(quad_params),
        controller=SE3Control(quad_params),
        trajectory=ThreeDCircularTraj(
            center=circle_center,
            radius=circle_radius,
            freq=circle_freq,
            yaw_traj=InwardYawFromTraj(circle_center, circle_radius, circle_freq)),
        wind_profile=None,
        world=world,
        camera=camera,          # OPTIONAL: onboard camera sensor
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

    # ------------------------------------------------------------------
    # 3. Run. The animation shows the live camera view next to the 3D scene,
    #    and the plots include sampled frames + visibility statistics.
    # ------------------------------------------------------------------
    print("Running simulation...")
    results = sim_instance.run(
        t_final=10,
        plot=True,
        plot_camera=True,
        plot_imu=False,
        plot_mocap=False,
        animate_bool=True,
        verbose=True,
        fname='camera_demo',   # saves camera_demo.mp4 when ffmpeg is available
    )

    # The captured frames live in results['camera_measurements'] with keys
    # time, image (K,H,W,3 uint8), visible_mask, projected, depth, keypoints,
    # keypoint_depths, visible_features. Save them separately from the csv.
    sim_instance.save_to_csv("camera_demo.csv")
    sim_instance.save_camera_data("rotorpy_demo", save_pngs=True)


if __name__ == "__main__":
    main()

from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from rotorpy.utils.camera_plotter import frustum_ray_directions, feature_colors

from rotorpy.utils.shapes import Quadrotor

import os

class ClosingFuncAnimation(FuncAnimation):
    def __init__(self, fig, func, *args, **kwargs):
        self._close_on_finish = kwargs.pop('close_on_finish')
        FuncAnimation.__init__(self, fig, func, *args, **kwargs)
        
    def _step(self, *args):
        still_going = FuncAnimation._step(self, *args)
        if self._close_on_finish and not still_going:
            plt.close(self._fig)

def _decimate_index(time, sample_time):
    """
    Given sorted lists of source times and sample times, return indices of
    source time closest to each sample time.
    """
    index = np.arange(time.size)
    sample_index = np.round(np.interp(sample_time, time, index)).astype(int)
    return sample_index

def animate(time, position, rotation, wind, animate_wind, world, filename=None, blit=False, show_axes=True, close_on_finish=False,
            camera_images=None, camera_times=None, camera=None, frustum_scale=None):
    """
    Animate a completed simulation result based on the time, position, and
    rotation history. The animation may be viewed live or saved to a .mp4 video
    (slower, requires additional libraries).

    For a live view, it is absolutely critical to retain a reference to the
    returned object in order to prevent garbage collection before the animation
    has completed displaying.

    Below, M corresponds to the number of drones you're animating. If M is None, i.e. the arrays are (N,3) and (N,3,3), then it is assumed that there is only one drone.
    Otherwise, we iterate over the M drones and animate them on the same axes.

    N is the number of time steps in the simulation.

    Parameters
        time, (N,) with uniform intervals
        position, (N,M,3)
        rotation, (N,M,3,3)
        wind, (N,M,3) world wind velocity
        animate_wind, if True animate wind vector
        world, a World object
        filename, for saved video, or live view if None
        blit, if True use blit for faster animation, default is False
        show_axes, if True plot axes, default is True
        close_on_finish, if True close figure at end of live animation or save, default is False
        camera_images, optional (K,H,W,3) uint8 array of camera frames to show in a panel next to the 3D view.
        camera_times, optional (K,) timestamps of the camera frames; each animation frame displays the most recent capture.
        camera, optional PinholeCamera object. When supplied together with
            camera_images/camera_times, the layout mirrors plot_camera_view():
            the 3D scene shows the surface features plus the camera frustum and
            triad moving with drone 0, next to the captured camera image.
        frustum_scale, length in meters of the camera frustum; None auto-scales
            to the world size.
    """

    # Check if there is only one drone.
    if len(position.shape) == 2:
        position = np.expand_dims(position, axis=1)
        rotation = np.expand_dims(rotation, axis=1)
        wind = np.expand_dims(wind, axis=1)
    M = position.shape[1]

    # Temporal style.
    rtf = 1.0 # real time factor > 1.0 is faster than real time playback
    render_fps = 30

    # Normalize the wind by the max of the wind magnitude on each axis, so that the maximum length of the arrow is decided by the scale factor
    wind_mag = np.max(np.linalg.norm(wind, axis=-1), axis=1)             # Get the wind magnitude time series
    max_wind = np.max(wind_mag)                         # Find the maximum wind magnitude in the time series

    if max_wind != 0:
        wind_arrow_scale_factor = 1                         # Scale factor for the wind arrow
        wind = wind_arrow_scale_factor*wind / max_wind

    # Decimate data to render interval; always include t=0.
    if time[-1] != 0:
        sample_time = np.arange(0, time[-1], 1/render_fps * rtf)
    else:
        sample_time = np.zeros((1,))
    index = _decimate_index(time, sample_time)
    time = time[index]
    position = position[index,:]
    rotation = rotation[index,:]
    wind = wind[index,:]

    # Set up axes.
    # When the camera view is shown next to the 3D scene, use a larger figure
    # so the two panels don't crowd each other.
    wide = camera_images is not None and camera_times is not None
    figsize = (12.0, 6.0) if wide else None
    if filename is not None:
        if isinstance(filename, Path):
            fig = plt.figure(filename.name, figsize=figsize)
        else:
            fig = plt.figure(filename, figsize=figsize)
    else:
        fig = plt.figure('Animation', figsize=figsize)
    fig.clear()
    if wide:
        # plt.figure() reuses existing figures and ignores figsize on reuse.
        fig.set_size_inches(12.0, 6.0, forward=True)
    fig.clear()

    # If camera frames are supplied, show them next to the 3D view. When the
    # camera object is also available, mirror plot_camera_view(): scatter the
    # surface features and draw the camera frustum/triad moving with drone 0.
    #
    # The frustum/triad are computed directly from the two transforms at hand:
    # the simulator's world-from-body pose (position + rotation matrices) and
    # the camera extrinsics (body -> camera). The extrinsics and corner-ray
    # directions are constant, so they are precomputed once below and each
    # frame only requires small matrix products; no quaternion conversions.
    cam_image_artist = None
    frustum_artists = []
    triad_artists = []
    _cam_data = None
    if camera_images is not None and camera_times is not None:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        ax_img = fig.add_subplot(gs[0, 1])
        cam_image_artist = ax_img.imshow(camera_images[0], origin='upper', interpolation='nearest')
        ax_img.set_title('Camera View')
        ax_img.set_axis_off()

        features = world.get_surface_features()
        if features is not None and len(features) > 0:
            c = feature_colors(world, len(features))
            ax.scatter(features[:, 0], features[:, 1], features[:, 2],
                       s=6, c=c, edgecolors='none', depthshade=False)

        if camera is not None:
            if frustum_scale is None:
                extents = np.asarray(world.world['bounds']['extents'])
                frustum_scale = 0.1 * np.max(extents[1::2] - extents[0::2])
            p_BC = np.asarray(camera.extrinsics['position'], dtype=float)          # body -> camera position
            R_BC_T = Rotation.from_quat(np.asarray(camera.extrinsics['orientation'],
                                                   dtype=float)).as_matrix().T      # camera -> body rotation
            ray_dirs = frustum_ray_directions(camera.intrinsics, frustum_scale) @ R_BC_T.T  # corner rays, body frame
            triad_dirs = R_BC_T.T                                                    # camera axes, body frame (rows)
            triad_scale = 0.5 * frustum_scale

            def camera_lines(f):
                """
                World-frame endpoints of the frustum rays/edges and triad arms
                for animation frame f.
                """
                R_WB = rotation[f, 0]                       # body -> world
                p_WC = position[f, 0] + R_WB @ p_BC         # camera origin in world
                corners = p_WC + ray_dirs @ R_WB.T          # image-plane corners in world
                arms = p_WC + triad_scale * (triad_dirs @ R_WB.T)  # camera x/y/z arms in world
                return (p_WC, corners, arms)

            _cam_data = camera_lines

            # Initial artists for frame 0: 4 corner rays + 4 rectangle edges,
            # then one arm per camera axis (x red, y green, z blue).
            (p_WC, corners, arms) = camera_lines(0)
            for c in corners:
                frustum_artists.append(ax.plot([p_WC[0], c[0]], [p_WC[1], c[1]], [p_WC[2], c[2]], color='k')[0])
            for i in range(4):
                a, b = corners[i], corners[(i + 1) % 4]
                frustum_artists.append(ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color='k')[0])
            for j, color in enumerate(['r', 'g', 'b']):
                e = arms[j]
                triad_artists.append(ax.plot([p_WC[0], e[0]], [p_WC[1], e[1]], [p_WC[2], e[2]], color=color)[0])
    else:
        ax = fig.add_subplot(projection='3d')

    if not show_axes:
        ax.set_axis_off()

    quads = [Quadrotor(ax, wind=animate_wind, wind_scale_factor=1) for _ in range(M)]

    world_artists = world.draw(ax)

    if cam_image_artist is not None:
        # The axes titles are used for the panel labels; keep time in a suptitle.
        title_artist = fig.suptitle('t = {}'.format(time[0]))
        ax.set_title('3D Scene')
    else:
        title_artist = ax.set_title('t = {}'.format(time[0]))

    def init():
        ax.draw(fig.canvas.get_renderer())
        # return world_artists + list(cquad.artists) + [title_artist]
        return world_artists + [title_artist] + [q.artists for q in quads]

    def update(frame):
        title_artist.set_text('t = {:.2f}'.format(time[frame]))
        for i, quad in enumerate(quads):
            quad.transform(position=position[frame,i,:], rotation=rotation[frame,i,:,:], wind=wind[frame,i,:])
        if cam_image_artist is not None:
            # Display the most recent camera frame captured at or before the
            # current animation time.
            cam_idx = int(np.clip(np.searchsorted(camera_times, time[frame], side='right') - 1,
                                  0, len(camera_times) - 1))
            cam_image_artist.set_data(camera_images[cam_idx])
        if frustum_artists:
            (p_WC, corners, arms) = _cam_data(frame)
            k = 0
            for c in corners:  # rays from the camera origin
                frustum_artists[k].set_data_3d([p_WC[0], c[0]], [p_WC[1], c[1]], [p_WC[2], c[2]])
                k += 1
            for i in range(4):  # image-plane rectangle edges
                a, b = corners[i], corners[(i + 1) % 4]
                frustum_artists[k].set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
                k += 1
            for j in range(3):  # camera x/y/z axes
                e = arms[j]
                triad_artists[j].set_data_3d([p_WC[0], e[0]], [p_WC[1], e[1]], [p_WC[2], e[2]])
        # [a.do_3d_projection(fig.canvas.get_renderer()) for a in quad.artists]   # No longer necessary in newer matplotlib?
        # return world_artists + list(quad.artists) + [title_artist]
        return world_artists + [title_artist] + [q.artists for q in quads]

    ani = ClosingFuncAnimation(fig=fig,
                        func=update,
                        frames=time.size,
                        init_func=init,
                        interval=1000.0/render_fps,
                        repeat=False,
                        blit=blit,
                        close_on_finish=close_on_finish)

    if filename is not None:
        print('Saving Animation')
        if not ".mp4" in filename:
            filename = filename + ".mp4"
        path = os.path.join(os.path.dirname(__file__),'..','data_out',filename)
        ani.save(path,
                 writer='ffmpeg',
                 fps=render_fps,
                 dpi=100)
        if close_on_finish:
            plt.close(fig)
            ani = None

    return ani
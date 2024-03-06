"""
TODO: Set up figure for appropriate target video size (eg. 720p).
TODO: Decide which additional user options should be available.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from rotorpy.utils.shapes import Quadrotor

import os

class ClosingFuncAnimation(FuncAnimation):
    def __init__(self, fig, func, *args, **kwargs):
        self._close_on_finish = kwargs.pop('close_on_finish')
        FuncAnimation.__init__(self, fig, func, *args, **kwargs)

    # def _stop(self, *args):
    #     super()._stop(self, *args)
    #     if self._close_on_finish:
    #         plt.close(self._fig)

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

def animate(time, position, rotation, wind, animate_wind, world, filename=None, blit=False, show_axes=True, close_on_finish=False):
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
    if filename is not None:
        if isinstance(filename, Path):
            fig = plt.figure(filename.name)
        else:
            fig = plt.figure(filename)
    else:
        fig = plt.figure('Animation')
    fig.clear()
    ax = fig.add_subplot(projection='3d')
    if not show_axes:
        ax.set_axis_off()

    quads = [Quadrotor(ax, wind=animate_wind, wind_scale_factor=1) for _ in range(M)]

    world_artists = world.draw(ax)

    title_artist = ax.set_title('t = {}'.format(time[0]))

    def init():
        ax.draw(fig.canvas.get_renderer())
        # return world_artists + list(cquad.artists) + [title_artist]
        return world_artists + [title_artist] + [q.artists for q in quads]

    def update(frame):
        title_artist.set_text('t = {:.2f}'.format(time[frame]))
        for i, quad in enumerate(quads):
            quad.transform(position=position[frame,i,:], rotation=rotation[frame,i,:,:], wind=wind[frame,i,:])
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

if __name__ == "__main__":

    from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.
    from rotorpy.learning.quadrotor_environments import QuadrotorEnv
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.trajectories.circular_traj import CircularTraj

    import gymnasium as gym

    output_video_dir = os.path.join(os.path.dirname(__file__), "..", "data_out", "test_animation.mp4")

    # Create M SE3 drones.
    M = 3
    baseline_controller = SE3Control(quad_params) 

    def make_env():
        return gym.make("Quadrotor-v0", 
                    control_mode ='cmd_motor_speeds', 
                    quad_params = quad_params,
                    max_time = 5,
                    world = None,
                    sim_rate = 100,
                    render_mode='3D',
                    render_fps = 60,
                    color='b')

    envs = [make_env() for _ in range(M)]

    # For each environment command it to do a circle with random radius and location. 
    trajs = []
    for env in envs:
        center = np.random.uniform(low=-2, high=2, size=3)
        radius = np.random.uniform(low=0.5, high=1.5)
        freq = np.random.uniform(low=0.1, high=0.3)
        plane = np.random.choice(['XY', 'YZ', 'XZ'])
        traj = CircularTraj(center=center, radius=radius, freq=freq, plane=plane, direction=np.random.choice(['CW', 'CCW']))
        trajs.append(traj)

    # Collect observations for each environment.
    observations = [env.reset(initial_state='random')[0] for env in envs]

    # This is a list of env termination conditions so that the loop only ends when the final env is terminated. 
    terminated = [False]*len(observations)

    # Arrays for animating. 
    T = [0]
    position = [[obs[0:3] for obs in observations]]
    quat = [[obs[6:10] for obs in observations]]

    while not all(terminated):
        for (i, env) in enumerate(envs):  # For each environment...
            # Unpack the observation from the environment
            state = {'x': observations[i][0:3], 'v': observations[i][3:6], 'q': observations[i][6:10], 'w': observations[i][10:13]}

            # Command the quad to do circles.
            flat = trajs[i].update(env.unwrapped.t)
            control_dict = baseline_controller.update(0, state, flat)

            # Extract the commanded motor speeds.
            cmd_motor_speeds = control_dict['cmd_motor_speeds']

            # The environment expects the control inputs to all be within the range [-1,1]
            action = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])

            # For the last environment, append the current timestep.
            if i == 0: 
                T.append(env.unwrapped.t)

            # Step the environment forward. 
            observations[i], reward, terminated[i], truncated, info = env.step(action)

        # Append arrays for plotting.
        position.append([obs[0:3] for obs in observations])
        quat.append([obs[6:10] for obs in observations])

    # Convert to numpy arrays.
    T = np.array(T)
    position = np.array(position)
    quat = np.array(quat)

    # Convert the quaternion to rotation matrix.
    rotation = np.array([Rotation.from_quat(quat[i]).as_matrix() for i in range(T.size)])

    # Animate the results.
    ani = animate(T, position, rotation, wind=np.zeros((T.size,M,3)), animate_wind=False, world=envs[0].world, filename=output_video_dir, blit=False, show_axes=True, close_on_finish=True)
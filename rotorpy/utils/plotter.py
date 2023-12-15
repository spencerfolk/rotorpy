import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# from rotorpy.utils.axes3ds import Axes3Ds
from rotorpy.utils.animate import animate

import os

"""
Functions for showing the results from the simulator.

"""

class Plotter():

    def __init__(self, results, world):

        (self.time, self.x, self.x_des, self.v, 
        self.v_des, self.q, self.q_des, self.w, 
        self.s, self.s_des, self.M, self.T, self.wind,
        self.accel, self.gyro, self.accel_gt,
        self.x_mc, self.v_mc, self.q_mc, self.w_mc, 
        self.filter_state, self.covariance, self.sd) = self.unpack_results(results)

        self.R = Rotation.from_quat(self.q).as_matrix()
        self.R_mc = Rotation.from_quat(self.q_mc).as_matrix() # Rotation as measured by motion capture.

        self.world = world

        return

    def plot_results(self, plot_mocap, plot_estimator, plot_imu):
        """
        Plot the results

        """

        # 3D Paths
        fig = plt.figure('3D Path')
        # ax = Axes3Ds(fig)
        ax = fig.add_subplot(projection='3d')
        self.world.draw(ax)
        ax.plot3D(self.x[:,0], self.x[:,1], self.x[:,2], 'b.')
        ax.plot3D(self.x_des[:,0], self.x_des[:,1], self.x_des[:,2], 'k')

        # Position and Velocity vs. Time
        (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Pos/Vel vs Time')
        ax = axes[0]
        ax.plot(self.time, self.x_des[:,0], 'r', self.time, self.x_des[:,1], 'g', self.time, self.x_des[:,2], 'b')
        ax.plot(self.time, self.x[:,0], 'r.',    self.time, self.x[:,1], 'g.',    self.time, self.x[:,2], 'b.')
        ax.legend(('x', 'y', 'z'))
        ax.set_ylabel('position, m')
        ax.grid('major')
        ax.set_title('Position')
        ax = axes[1]
        ax.plot(self.time, self.v_des[:,0], 'r', self.time, self.v_des[:,1], 'g', self.time, self.v_des[:,2], 'b')
        ax.plot(self.time, self.v[:,0], 'r.',    self.time, self.v[:,1], 'g.',    self.time, self.v[:,2], 'b.')
        ax.legend(('x', 'y', 'z'))
        ax.set_ylabel('velocity, m/s')
        ax.set_xlabel('time, s')
        ax.grid('major')

        # Orientation and Angular Velocity vs. Time
        (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Attitude/Rate vs Time')
        ax = axes[0]
        ax.plot(self.time, self.q_des[:,0], 'r', self.time, self.q_des[:,1], 'g', self.time, self.q_des[:,2], 'b', self.time, self.q_des[:,3], 'm')
        ax.plot(self.time, self.q[:,0], 'r.',    self.time, self.q[:,1], 'g.',    self.time, self.q[:,2], 'b.',    self.time, self.q[:,3],     'm.')
        ax.legend(('i', 'j', 'k', 'w'))
        ax.set_ylabel('quaternion')
        ax.set_xlabel('time, s')
        ax.grid('major')
        ax = axes[1]
        ax.plot(self.time, self.w[:,0], 'r.', self.time, self.w[:,1], 'g.', self.time, self.w[:,2], 'b.')
        ax.legend(('x', 'y', 'z'))
        ax.set_ylabel('angular velocity, rad/s')
        ax.set_xlabel('time, s')
        ax.grid('major')

        if plot_mocap:  # if mocap should be plotted. 
            # Motion capture position and velocity vs time
            (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Motion Capture Pos/Vel vs Time')
            ax = axes[0]
            ax.plot(self.time, self.x_mc[:,0], 'r.', self.time, self.x_mc[:,1], 'g.',    self.time, self.x_mc[:,2], 'b.')
            ax.legend(('x', 'y', 'z'))
            ax.set_ylabel('position, m')
            ax.grid('major')
            ax.set_title('MOTION CAPTURE Position/Velocity')
            ax = axes[1]
            ax.plot(self.time, self.v_mc[:,0], 'r.',    self.time, self.v_mc[:,1], 'g.',    self.time, self.v_mc[:,2], 'b.')
            ax.legend(('x', 'y', 'z'))
            ax.set_ylabel('velocity, m/s')
            ax.set_xlabel('time, s')
            ax.grid('major')
            # Motion Capture Orientation and Angular Velocity vs. Time
            (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Motion Capture Attitude/Rate vs Time')
            ax = axes[0]
            ax.plot(self.time, self.q_mc[:,0], 'r.',    self.time, self.q_mc[:,1], 'g.',    self.time, self.q_mc[:,2], 'b.',    self.time, self.q_mc[:,3],     'm.')
            ax.legend(('i', 'j', 'k', 'w'))
            ax.set_ylabel('quaternion')
            ax.set_xlabel('time, s')
            ax.grid('major')
            ax.set_title("MOTION CAPTURE Attitude/Rate")
            ax = axes[1]
            ax.plot(self.time, self.w_mc[:,0], 'r.', self.time, self.w_mc[:,1], 'g.', self.time, self.w_mc[:,2], 'b.')
            ax.legend(('x', 'y', 'z'))
            ax.set_ylabel('angular velocity, rad/s')
            ax.set_xlabel('time, s')
            ax.grid('major')

        # Commands vs. Time
        (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
        ax = axes[0]
        ax.plot(self.time, self.s_des[:,0], 'r', self.time, self.s_des[:,1], 'g', self.time, self.s_des[:,2], 'b',  self.time, self.s_des[:,3], 'k')
        ax.plot(self.time, self.s[:,0], 'r.',    self.time, self.s[:,1], 'g.',    self.time, self.s[:,2], 'b.',     self.time, self.s[:,3], 'k.')
        ax.legend(('1', '2', '3', '4'))
        ax.set_ylabel('motor speeds, rad/s')
        ax.grid('major')
        ax.set_title('Commands')
        ax = axes[1]
        ax.plot(self.time, self.M[:,0], 'r.', self.time, self.M[:,1], 'g.', self.time, self.M[:,2], 'b.')
        ax.legend(('x', 'y', 'z'))
        ax.set_ylabel('moment, N*m')
        ax.grid('major')
        ax = axes[2]
        ax.plot(self.time, self.T, 'k.')
        ax.set_ylabel('thrust, N')
        ax.set_xlabel('time, s')
        ax.grid('major')

        # Winds
        (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Winds vs Time')
        ax = axes[0]
        ax.plot(self.time, self.wind[:,0], 'r')
        ax.set_ylabel("wind X, m/s")
        ax.grid('major')
        ax.set_title('Winds')
        ax = axes[1]
        ax.plot(self.time, self.wind[:,1], 'g')
        ax.set_ylabel("wind Y, m/s")
        ax.grid('major')
        ax = axes[2]
        ax.plot(self.time, self.wind[:,2], 'b')
        ax.set_ylabel("wind Z, m/s")
        ax.set_xlabel("time, s")
        ax.grid('major')

        # IMU sensor
        if plot_imu:
            (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num="IMU Measurements vs Time")
            ax = axes[0]
            ax.plot(self.time, self.accel[:,0], 'r.', self.time, self.accel[:,1], 'g.', self.time, self.accel[:,2], 'b.')
            ax.plot(self.time, self.accel_gt[:,0], 'k', self.time, self.accel_gt[:,1], 'c', self.time, self.accel_gt[:,2], 'm')
            ax.set_ylabel("linear acceleration, m/s/s")
            ax.grid()
            ax = axes[1]
            ax.plot(self.time, self.gyro[:,0], 'r.', self.time, self.gyro[:,1], 'g.', self.time, self.gyro[:,2], 'b.')
            ax.set_ylabel("angular velocity, rad/s")
            ax.grid()
            ax.legend(('x','y','z'))
            ax.set_xlabel("time, s")

        if plot_estimator:
            if self.estimator_exists:
                N_filter = self.filter_state.shape[1]
                (fig, axes) = plt.subplots(nrows=N_filter, ncols=1, sharex=True, num="Filter States vs Time")
                fig.set_size_inches(11, 8.5)
                for i in range(N_filter):
                    ax = axes[i]
                    ax.plot(self.time, self.filter_state[:,i], 'k', )
                    ax.fill_between(self.time, self.filter_state[:,i]-self.sd[:,i], self.filter_state[:,i]+self.sd[:,i], alpha=0.3, color='k')
                    ax.set_ylabel("x"+str(i))
                ax.set_xlabel("Time, s")

                (fig, axes) = plt.subplots(nrows=N_filter, ncols=1, sharex=True, num="Filter Covariance vs Time")
                fig.set_size_inches(11, 8.5)
                for i in range(N_filter):
                    ax = axes[i]
                    ax.plot(self.time, self.sd[:,i]**2, 'k', )
                    ax.set_ylabel("cov(x"+str(i)+")")
                ax.set_xlabel("Time, s")

        plt.show()

        return

    def animate_results(self, animate_wind, fname=None):
        """
        Animate the results
        
        """

        # Animation (Slow)
        # Instead of viewing the animation live, you may provide a .mp4 filename to save.
        ani = animate(self.time, self.x, self.R, self.wind, animate_wind, world=self.world, filename=fname)
        plt.show()

        return

    def unpack_results(self, result):

        # Unpack the dictionary of results
        time                = result['time']
        state               = result['state']
        control             = result['control']
        flat                = result['flat']
        imu_measurements    = result['imu_measurements']
        imu_gt              = result['imu_gt']
        mocap               = result['mocap_measurements']
        state_estimate      = result['state_estimate']

        # Unpack each result into NumPy arrays
        x = state['x']
        x_des = flat['x']
        v = state['v']
        v_des = flat['x_dot']

        q = state['q']
        q_des = control['cmd_q']
        w = state['w']

        s_des = control['cmd_motor_speeds']
        s = state['rotor_speeds']
        M = control['cmd_moment']
        T = control['cmd_thrust']

        wind = state['wind']

        accel   = imu_measurements['accel']
        gyro    = imu_measurements['gyro']

        accel_gt = imu_gt['accel']

        x_mc = mocap['x']
        v_mc = mocap['v']
        q_mc = mocap['q']
        w_mc = mocap['w']

        filter_state = state_estimate['filter_state']
        covariance = state_estimate['covariance']
        if filter_state.shape[1] > 0:
            sd = 3*np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
            self.estimator_exists = True
        else:
            sd = []
            self.estimator_exists = False

        return (time, x, x_des, v, v_des, q, q_des, w, s, s_des, M, T, wind, accel, gyro, accel_gt, x_mc, v_mc, q_mc, w_mc, filter_state, covariance, sd)

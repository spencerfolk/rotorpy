import numpy as np
import sys

class CircularTraj(object):
    """
    A circle. 
    """
    def __init__(self, center=np.array([0,0,0]), radius=1, freq=0.2, yaw_bool=False, plane='XY', direction='CCW'):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.

        Inputs:
            center, the center of the circle (m)
            radius, the radius of the circle (m)
            freq, the frequency with which a circle is completed (Hz)
            yaw_bool, determines if yaw motion is desired
            plane, the plane with which the circle lies on, 'XY', 'YZ', or 'XZ' 
            direcition, the direction of the circle, 'CCW' or 'CW'
        """

        # Check and assign inputs
        if plane == "XY" or plane == "YZ" or plane == "XZ":
            self.plane = plane
        else:
            print("CircularTraj Error: incorrect specification of plane. Must be 'XY', 'YZ', or 'XZ' ")
            sys.exit(1)

        if direction == "CW" or direction == "CCW":
            if direction == "CW":
                self.sign = -1
            else:
                self.sign = 1
        else:
            print("CircularTraj Error: incorrect specification of direction. Must be 'CW' or 'CCW' ")
            sys.exit(1)

        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.radius = radius
        self.freq = freq

        self.omega = 2*np.pi*self.freq
        self.yaw_bool = yaw_bool


    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """

        if self.plane == "XY":
            x        = np.array([self.cx + self.radius*np.cos(self.sign*self.omega*t),
                                self.cy + self.radius*np.sin(self.sign*self.omega*t),
                                self.cz])
            x_dot    = np.array([-self.radius*self.sign*self.omega*np.sin(self.sign*self.omega*t),
                                self.radius*self.sign*self.omega*np.cos(self.sign*self.omega*t),
                                0])
            x_ddot   = np.array([-self.radius*((self.sign*self.omega)**2)*np.cos(self.sign*self.omega*t),
                                -self.radius*((self.sign*self.omega)**2)*np.sin(self.sign*self.omega*t),
                                0])
            x_dddot  = np.array([self.radius*((self.sign*self.omega)**3)*np.sin(self.sign*self.omega*t),
                                -self.radius*((self.sign*self.omega)**3)*np.cos(self.sign*self.omega*t),
                                0])
            x_ddddot = np.array([self.radius*((self.sign*self.omega)**4)*np.cos(self.sign*self.omega*t),
                                self.radius*((self.sign*self.omega)**4)*np.sin(self.sign*self.omega*t),
                                0])
        elif self.plane == "YZ":
            x        = np.array([self.cx,
                                self.cy + self.radius*np.cos(self.sign*self.omega*t),
                                self.cz + self.radius*np.sin(self.sign*self.omega*t)])
            x_dot    = np.array([0,
                                -self.radius*self.sign*self.omega*np.sin(self.sign*self.omega*t),
                                self.radius*self.sign*self.omega*np.cos(self.sign*self.omega*t)])
            x_ddot   = np.array([0,
                                -self.radius*((self.sign*self.omega)**2)*np.cos(self.sign*self.omega*t),
                                -self.radius*((self.sign*self.omega)**2)*np.sin(self.sign*self.omega*t)])
            x_dddot  = np.array([0,
                                self.radius*((self.sign*self.omega)**3)*np.sin(self.sign*self.omega*t),
                                -self.radius*((self.sign*self.omega)**3)*np.cos(self.sign*self.omega*t)])
            x_ddddot = np.array([0,
                                self.radius*((self.sign*self.omega)**4)*np.cos(self.sign*self.omega*t),
                                self.radius*((self.sign*self.omega)**4)*np.sin(self.sign*self.omega*t)])
        elif self.plane == "XZ":
            x        = np.array([self.cx + self.radius*np.cos(self.sign*self.omega*t),
                                self.cy,
                                self.cz + self.radius*np.sin(self.sign*self.omega*t)])
            x_dot    = np.array([-self.radius*self.sign*self.omega*np.sin(self.sign*self.omega*t),
                                0,
                                self.radius*self.sign*self.omega*np.cos(self.sign*self.omega*t)])
            x_ddot   = np.array([-self.radius*((self.sign*self.omega)**2)*np.cos(self.sign*self.omega*t),
                                0,
                                -self.radius*((self.sign*self.omega)**2)*np.sin(self.sign*self.omega*t)])
            x_dddot  = np.array([self.radius*((self.sign*self.omega)**3)*np.sin(self.omega*t),
                                0,
                                -self.radius*((self.sign*self.omega)**3)*np.cos(self.sign*self.omega*t)])
            x_ddddot = np.array([self.radius*((self.sign*self.omega)**4)*np.cos(self.sign*self.omega*t),
                                0,
                                self.radius*((self.sign*self.omega)**4)*np.sin(self.sign*self.omega*t)])

        if self.yaw_bool:
            yaw = np.pi/4*np.sin(np.pi*t)
            yaw_dot = np.pi*np.pi/4*np.cos(np.pi*t)
            yaw_ddot = -np.pi*np.pi*np.pi/4*np.sin(np.pi*t)

        else:
            yaw = 0
            yaw_dot = 0
            yaw_ddot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot, 'yaw_ddot':yaw_ddot}
        return flat_output

class ThreeDCircularTraj(object):
    """

    """
    def __init__(self, center=np.array([0,0,0]), radius=np.array([1,1,1]), freq=np.array([0.2,0.2,0.2]), yaw_bool=False):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.

        Inputs:
            center, the center of the circle (m)
            radius, the radius of the circle (m)
            freq, the frequency with which a circle is completed (Hz)
        """

        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.radius = radius
        self.freq = freq

        self.omega = 2*np.pi*self.freq

        self.yaw_bool = yaw_bool

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.array([self.cx + self.radius[0]*np.cos(self.omega[0]*t),
                             self.cy + self.radius[1]*np.sin(self.omega[1]*t),
                             self.cz + self.radius[2]*np.sin(self.omega[2]*t)])
        x_dot    = np.array([-self.radius[0]*self.omega[0]*np.sin(self.omega[0]*t),
                              self.radius[1]*self.omega[1]*np.cos(self.omega[1]*t),
                              self.radius[2]*self.omega[2]*np.cos(self.omega[2]*t)])
        x_ddot   = np.array([-self.radius[0]*(self.omega[0]**2)*np.cos(self.omega[0]*t),
                             -self.radius[1]*(self.omega[1]**2)*np.sin(self.omega[1]*t),
                             -self.radius[2]*(self.omega[2]**2)*np.sin(self.omega[2]*t)])
        x_dddot  = np.array([ self.radius[0]*(self.omega[0]**3)*np.sin(self.omega[0]*t),
                             -self.radius[1]*(self.omega[1]**3)*np.cos(self.omega[1]*t),
                              self.radius[2]*(self.omega[2]**3)*np.cos(self.omega[2]*t)])
        x_ddddot = np.array([self.radius[0]*(self.omega[0]**4)*np.cos(self.omega[0]*t),
                             self.radius[1]*(self.omega[1]**4)*np.sin(self.omega[1]*t),
                             self.radius[2]*(self.omega[2]**4)*np.sin(self.omega[2]*t)])

        if self.yaw_bool:
            yaw = 0.8*np.pi/2*np.sin(2.5*t)
            yaw_dot = 0.8*2.5*np.pi/2*np.cos(2.5*t)
        else:
            yaw = 0
            yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
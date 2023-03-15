import numpy as np
"""
Lissajous curves are defined by trigonometric functions parameterized in time. 
See https://en.wikipedia.org/wiki/Lissajous_curve

"""
class TwoDLissajous(object):
    """
    The standard Lissajous on the XY curve as defined by https://en.wikipedia.org/wiki/Lissajous_curve
    This is planar in the XY plane at a fixed height. 
    """
    def __init__(self, A=1, B=1, a=1, b=1, delta=0, height=0, yaw_bool=False):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.

        Inputs:
            A := amplitude on the X axis
            B := amplitude on the Y axis
            a := frequency on the X axis
            b := frequency on the Y axis
            delta := phase offset between the x and y parameterization
            height := the z height that the lissajous occurs at
            yaw_bool := determines whether the vehicle should yaw
        """

        self.A, self.B = A, B
        self.a, self.b = a, b 
        self.delta = delta
        self.height = height

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
        x        = np.array([self.A*np.sin(self.a*t + self.delta),
                             self.B*np.sin(self.b*t),
                             self.height])
        x_dot    = np.array([self.a*self.A*np.cos(self.a*t + self.delta),
                             self.b*self.B*np.cos(self.b*t),
                             0])
        x_ddot   = np.array([-(self.a)**2*self.A*np.sin(self.a*t + self.delta),
                             -(self.b)**2*self.B*np.sin(self.b*t),
                             0])
        x_dddot  = np.array([-(self.a)**3*self.A*np.cos(self.a*t + self.delta),
                             -(self.b)**3*self.B*np.cos(self.b*t),
                             0])
        x_ddddot = np.array([(self.a)**4*self.A*np.sin(self.a*t + self.delta),
                             (self.b)**4*self.B*np.sin(self.b*t),
                             0])

        if self.yaw_bool:
            yaw = np.pi/4*np.sin(np.pi*t)
            yaw_dot = np.pi*np.pi/4*np.cos(np.pi*t)
        else:
            yaw = 0
            yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

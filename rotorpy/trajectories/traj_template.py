"""
Imports
"""
import numpy as np

class TrajTemplate(object):
    """
    The trajectory is implemented as a class. There are two required methods for each trajectory class: __init__() and update().
    The __init__() is required for instantiating the class with appropriate parameterizations. For example, if you are doing 
    a circular trajectory you might want to specify the radius of the circle. 
    The update() method is called at each iteration of the simulator. The only input to update is time t. The output of update()
    should be the desired flat outputs in a dictionary, as specified below. 
    """
    def __init__(self):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.
        """

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
        x    = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw    = 0
        yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

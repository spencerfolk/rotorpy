"""
Yaw trajectory controllers.

These are not full flat-output trajectories; they only produce yaw, yaw_dot,
and yaw_ddot. They are passed to any trajectory's ``yaw_traj`` parameter to
override the default zero-yaw behavior.
"""
import numpy as np


class ForwardYaw:
    """
    Yaw trajectory that makes the drone face in the direction of travel.

    Computes yaw from the velocity vector in the xy-plane: yaw = arctan2(vy, vx).
    When the horizontal speed is below a threshold, the last valid yaw is held
    constant to avoid flickering.

    Usage:
        yaw_traj = ForwardYaw()
        # Then pass yaw_traj=... to any trajectory's __init__.
    """
    def __init__(self, speed_threshold=1e-2):
        """
        Parameters:
            speed_threshold, minimum horizontal speed (m/s) below which the
                previous yaw is held constant. Default is 0.01.
        """
        self.speed_threshold = speed_threshold
        self._prev_yaw = 0.0
        self._prev_yaw_raw = 0.0
        self._prev_t = None

    def update(self, t, x_dot=None):
        """
        Compute the desired yaw from the velocity vector.

        Inputs:
            t, current time (s)
            x_dot, velocity vector (3,), or None (defaults to zero velocity)

        Outputs:
            yaw_dict, dict with keys
                yaw,     desired yaw angle (rad)
                yaw_dot, desired yaw rate (rad/s)
                yaw_ddot, desired yaw acceleration (rad/s^2)
        """
        if x_dot is None:
            x_dot = np.zeros(3)

        xy_speed = np.sqrt(x_dot[0]**2 + x_dot[1]**2)

        if xy_speed > self.speed_threshold:
            yaw = np.arctan2(x_dot[1], x_dot[0])
            self._prev_yaw = yaw
        else:
            yaw = self._prev_yaw

        # Finite-difference derivatives with angle-wraparound handling.
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

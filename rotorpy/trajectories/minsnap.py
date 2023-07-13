"""
Imports
"""
import numpy as np
import cvxopt
from scipy.linalg import block_diag

def cvxopt_solve_qp(P, q=None, G=None, h=None, A=None, b=None):
    """
    From https://scaron.info/blog/quadratic-programming-in-python.html . Infrastructure code for solving quadratic programs using CVXOPT. 
    The structure of the program is as follows: 

    min 0.5 xT P x + qT x
    s.t. Gx <= h
         Ax = b
    Inputs:
        P, numpy array, the quadratic term of the cost function
        q, numpy array, the linear term of the cost function
        G, numpy array, inequality constraint matrix
        h, numpy array, inequality constraint vector
        A, numpy array, equality constraint matrix
        b, numpy array, equality constraint vector
    Outputs:
        The optimal solution to the quadratic program
    """
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def H_fun(dt):
    """
    Computes the cost matrix for a single segment in a single dimension.
    *** Assumes that the decision variables c_i are e.g. x(t) = c_0 + c_1*t + c_2*t^2 + c_3*t^3 + c_4*t^4 + c_5*t^5 + c_6*t^6 + c_7*t^7
    Inputs:
        dt, scalar, the duration of the segment (t_(i+1) - t_i) 
    Outputs:
        H, numpy array, matrix containing the min snap cost function for that segment. Assumes the polynomial is of order 7.
    """

    cost = np.array([[576*dt, 1440*dt**2, 2880*dt**3, 5040*dt**4],
                     [1440*dt**2, 4800*dt**3, 10800*dt**4, 20160*dt**5],
                     [2880*dt**3, 10800*dt**4, 25920*dt**5, 50400*dt**6],
                     [5040*dt**4, 20160*dt**5, 50400*dt**6, 100800*dt**7]])

    H = np.zeros((8,8))
    H[4:,4:] = cost

    return H 

def get_1d_equality_constraints(keyframes, delta_t, m, vmax=2):
    """
    Computes the equality constraints for the min snap problem. 
    *** Assumes that the decision variables c_i are e.g. x(t) = c_0 + c_1*t + c_2*t^2 + c_3*t^3 + c_4*t^4 + c_5*t^5 + c_6*t^6 + c_7*t^7

    Inputs:
        keyframes, numpy array, a list of m waypoints IN ONE DIMENSION (x,y,z, or yaw)

    """
    # N = keyframes.shape[0]  # the number of keyframes
    K = 8*m                 # the number of decision variables

    A = np.zeros((8*m, K))
    b = np.zeros((8*m,))

    G = np.zeros((m, K))
    h = np.zeros((m))


    for i in range(m): # for each segment...
        # Compute the segment duration
        dt = delta_t[i]

        # Position continuity at the beginning of the segment
        A[i, 8*i:(8*i+8)] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        b[i] = keyframes[i]

        # Position continuity at the end of the segment
        A[m+i, 8*i:(8*i+8)] = np.array([1, dt, dt**2, dt**3, dt**4, dt**5, dt**6, dt**7])
        b[m+i] = keyframes[i+1]

        ###### At this point we have 2*m constraints.. 

        # Intermediate smoothness constraints
        if i < (m-1): # we don't want to include the last segment for this loop
            A[2*m + 6*i + 0, 8*i:(8*i+16)] = np.array([0, -1, -2*dt, -3*dt**2, -4*dt**3, -5*dt**4, -6*dt**5, -7*dt**6, 0, 1, 0, 0, 0, 0, 0, 0])     # Velocity
            A[2*m + 6*i + 1, 8*i:(8*i+16)] = np.array([0, 0, -2, -6*dt, -12*dt**2, -20*dt**3, -30*dt**4, -42*dt**5, 0, 0, 2, 0, 0, 0, 0, 0])        # Acceleration
            A[2*m + 6*i + 2, 8*i:(8*i+16)] = np.array([0, 0, 0, -6, -24*dt, -60*dt**2, -120*dt**3, -210*dt**4, 0, 0, 0, 6, 0, 0, 0, 0])             # Jerk
            A[2*m + 6*i + 3, 8*i:(8*i+16)] = np.array([0, 0, 0, 0, -24, -120*dt, -360*dt**2, -840*dt**3, 0, 0, 0, 0, 24, 0, 0, 0])                  # Snap
            A[2*m + 6*i + 4, 8*i:(8*i+16)] = np.array([0, 0, 0, 0, 0, -120, -720*dt, -2520*dt**2, 0, 0, 0, 0, 0, 120, 0, 0])                        # 5TH derivative
            A[2*m + 6*i + 5, 8*i:(8*i+16)] = np.array([0, 0, 0, 0, 0, 0, -720, -5040*dt, 0, 0, 0, 0, 0, 0, 720, 0])                                 # 6TH derivative

            b[2*m + 6*i + 0] = 0
            b[2*m + 6*i + 1] = 0
            b[2*m + 6*i + 2] = 0
            b[2*m + 6*i + 3] = 0
            b[2*m + 6*i + 4] = 0
            b[2*m + 6*i + 5] = 0

        G[i, 8*i:(8*i + 8)] = np.array([0, 1, 2*(0.5*dt), 3*(0.5*dt)**2, 4*(0.5*dt)**3, 5*(0.5*dt)**4, 6*(0.5*dt)**5, 7*(0.5*dt)**6])
        h[i] = vmax

    # Now we have added an addition 6*(m-1) constraints, total 2*m + 6*(m-1) = 2m + 8m - 6 = 8m - 6 constraints
    # Last constraints are on the first and last trajectory segments. Higher derivatives are = 0
    A[8*m - 6, 0:8] = np.array([0, 1, 0, 0, 0, 0, 0, 0])                                    # Velocity = 0 at start
    b[8*m - 6] = 0
    A[8*m - 5, -8:] = np.array([0, 1, 2*dt, 3*dt**2, 4*dt**3, 5*dt**4, 6*dt**5, 7*dt**6])   # Velocity = 0 at end
    b[8*m - 5] = 0
    A[8*m - 4, 0:8] = np.array([0, 0, 2, 0, 0, 0, 0, 0])                                    # Acceleration = 0 at start
    b[8*m - 4] = 0
    A[8*m - 3, -8:] = np.array([0, 0, 2, 6*dt,12*dt**2, 20*dt**3, 30*dt**4, 42*dt**5])      # Acceleration = 0 at end
    b[8*m - 3] = 0
    A[8*m - 2, 0:8] = np.array([0, 0, 0, 6, 0, 0, 0, 0])                                    # Jerk = 0 at start
    b[8*m - 2] = 0
    A[8*m - 1, -8:] = np.array([0, 0, 0, 6, 24*dt, 60*dt**2, 120*dt**3, 210*dt**4])         # Jerk = 0 at end
    b[8*m - 1] = 0

    return (A, b, G, h)

class MinSnap(object):
    """
    MinSnap generates a minimum snap trajectory for the quadrotor, following https://ieeexplore.ieee.org/document/5980409. 
    The trajectory is a piecewise 7th order polynomial (minimum degree necessary for snap optimality). 
    """
    def __init__(self, points, yaw_angles=None, v_avg=2):
        """
        Waypoints and yaw angles compose the "keyframes" for optimizing over. 
        Inputs:
            points, numpy array of m 3D waypoints. 
            yaw_angles, numpy array of m yaw angles corresponding to each waypoint. 
            v_avg, the average speed between waypoints, this is used to do the time allocation. 
        """

        if yaw_angles is None:
            self.yaw = np.zeros((points.shape[0]))
        else:
            self.yaw = yaw_angles
        self.v_avg = v_avg

        # Compute the distances between each waypoint.
        seg_dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
        seg_mask = np.append(True, seg_dist > 1e-3)
        self.points = points[seg_mask,:]

        self.null = False

        m = self.points.shape[0]-1  # Get the number of segments

        # Compute the derivatives of the polynomials
        self.x_dot_poly    = np.zeros((m, 3, 7))
        self.x_ddot_poly   = np.zeros((m, 3, 6))
        self.x_dddot_poly  = np.zeros((m, 3, 5))
        self.x_ddddot_poly = np.zeros((m, 3, 4))
        self.yaw_dot_poly = np.zeros((m, 1, 7))
        self.yaw_ddot_poly = np.zeros((m, 1, 6))
        
        # If two or more waypoints remain, solve min snap
        if self.points.shape[0] >= 2:

            ################## Time allocation
            self.delta_t = seg_dist/self.v_avg # Compute the segment durations based on the average velocity
            self.t_keyframes = np.concatenate(([0], np.cumsum(self.delta_t)))  # Construct time array which indicates when the quad should be at the i'th waypoint. 

            ################## Cost function
            # First get the cost segment for each matrix: 
            H = [H_fun(self.delta_t[i]) for i in range(m)]

            # Now concatenate these costs using block diagonal form:
            P = block_diag(*H)

            # Lastly the linear term in the cost function is 0
            q = np.zeros((8*m,1))
            
            ################## Constraints for each axis
            (Ax,bx,Gx,hx) = get_1d_equality_constraints(self.points[:,0], self.delta_t, m)
            (Ay,by,Gy,hy) = get_1d_equality_constraints(self.points[:,1], self.delta_t, m)
            (Az,bz,Gz,hz) = get_1d_equality_constraints(self.points[:,2], self.delta_t, m)
            (Ayaw,byaw,Gyaw,hyaw) = get_1d_equality_constraints(self.yaw, self.delta_t, m)

            ################## Solve for x, y, z, and yaw
            c_opt_x = np.linalg.solve(Ax,bx)
            c_opt_y = np.linalg.solve(Ay,by)
            c_opt_z = np.linalg.solve(Az,bz)
            c_opt_yaw = np.linalg.solve(Ayaw,byaw)

            ################## Construct polynomials from c_opt
            self.x_poly = np.zeros((m, 3, 8))
            self.yaw_poly = np.zeros((m, 1, 8))
            for i in range(m):
                self.x_poly[i,0,:] = np.flip(c_opt_x[8*i:(8*i+8)])
                self.x_poly[i,1,:] = np.flip(c_opt_y[8*i:(8*i+8)])
                self.x_poly[i,2,:] = np.flip(c_opt_z[8*i:(8*i+8)])
                self.yaw_poly[i,0,:] = np.flip(c_opt_yaw[8*i:(8*i+8)])

            for i in range(m):
                for j in range(3):
                    self.x_dot_poly[i,j,:]    = np.polyder(self.x_poly[i,j,:], m=1)
                    self.x_ddot_poly[i,j,:]   = np.polyder(self.x_poly[i,j,:], m=2)
                    self.x_dddot_poly[i,j,:]  = np.polyder(self.x_poly[i,j,:], m=3)
                    self.x_ddddot_poly[i,j,:] = np.polyder(self.x_poly[i,j,:], m=4)
                self.yaw_dot_poly[i,0,:] = np.polyder(self.yaw_poly[i,0,:], m=1)
                self.yaw_ddot_poly[i,0,:] = np.polyder(self.yaw_poly[i,0,:], m=2)

        else:
            # Otherwise, there is only one waypoint so we just set everything = 0. 
            self.null = True
            m = 1
            self.T = np.zeros((m,))
            self.x_poly = np.zeros((m, 3, 6))
            self.x_poly[0,:,-1] = points[0,:]

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
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw    = 0
        yaw_dot = 0
        yaw_ddot = 0

        if self.null:
            # If there's only one waypoint
            x    = self.points[0,:]
            yaw = self.yaw[0]
        else:
            # Find interval index i and time within interval t.
            t = np.clip(t, self.t_keyframes[0], self.t_keyframes[-1])
            for i in range(self.t_keyframes.size-1):
                if self.t_keyframes[i] + self.delta_t[i] >= t:
                    break
            t = t - self.t_keyframes[i]

            # Evaluate polynomial.
            for j in range(3):
                x[j]        = np.polyval(       self.x_poly[i,j,:], t)
                x_dot[j]    = np.polyval(   self.x_dot_poly[i,j,:], t)
                x_ddot[j]   = np.polyval(  self.x_ddot_poly[i,j,:], t)
                x_dddot[j]  = np.polyval( self.x_dddot_poly[i,j,:], t)
                x_ddddot[j] = np.polyval(self.x_ddddot_poly[i,j,:], t)

            yaw = np.polyval(self.yaw_poly[i, 0, :], t)
            yaw_dot = np.polyval(self.yaw_dot_poly[i,0,:], t)
            yaw_ddot = np.polyval(self.yaw_ddot_poly[i,0,:], t)

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot, 'yaw_ddot':yaw_ddot}
        return flat_output

if __name__=="__main__":

    import matplotlib.pyplot as plt
    from matplotlib import cm

    waypoints = np.array([[0,0,0],
                          [1,0,0],
                          [1,1,0],
                          [0,1,0],
                          [0,2,0],
                          [2,2,0]
                          ])
    yaw_angles = np.array([0, np.pi/2, 0, np.pi/4, 3*np.pi/2, 0])

    traj = MinSnap(waypoints, yaw_angles, v_avg=2)

    N = 1000
    time = np.linspace(0,5,N)
    x = np.zeros((N, 3))
    xdot = np.zeros((N,3))
    xddot = np.zeros((N,3))
    xdddot = np.zeros((N,3))
    xddddot = np.zeros((N,3))
    yaw = np.zeros((N,))
    yaw_dot = np.zeros((N,))

    for i in range(N):
        flat = traj.update(time[i])

        x[i,:] = flat['x']
        xdot[i,:] = flat['x_dot']
        xddot[i,:] = flat['x_ddot']
        xdddot[i,:] = flat['x_dddot']
        xddddot[i,:] = flat['x_ddddot']

        yaw[i] = flat['yaw']
        yaw_dot[i] = flat['yaw_dot']


    (fig, axes) = plt.subplots(nrows=5, ncols=1, sharex=True, num="Translational Flat Outputs")
    ax = axes[0]
    ax.plot(time, x[:,0], 'r-', label="X")
    ax.plot(time, x[:,1], 'g-', label="Y")
    ax.plot(time, x[:,2], 'b-', label="Z")
    ax.legend()
    ax.set_ylabel("x")
    ax = axes[1]
    ax.plot(time, xdot[:,0], 'r-', label="X")
    ax.plot(time, xdot[:,1], 'g-', label="Y")
    ax.plot(time, xdot[:,2], 'b-', label="Z")
    ax.set_ylabel("xdot")
    ax = axes[2]
    ax.plot(time, xddot[:,0], 'r-', label="X")
    ax.plot(time, xddot[:,1], 'g-', label="Y")
    ax.plot(time, xddot[:,2], 'b-', label="Z")
    ax.set_ylabel("xddot")
    ax = axes[3]
    ax.plot(time, xdddot[:,0], 'r-', label="X")
    ax.plot(time, xdddot[:,1], 'g-', label="Y")
    ax.plot(time, xdddot[:,2], 'b-', label="Z")
    ax.set_ylabel("xdddot")
    ax = axes[4]
    ax.plot(time, xddddot[:,0], 'r-', label="X")
    ax.plot(time, xddddot[:,1], 'g-', label="Y")
    ax.plot(time, xddddot[:,2], 'b-', label="Z")
    ax.set_ylabel("xddddot")
    ax.set_xlabel("Time, s")

    (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num="Yaw vs Time")
    ax = axes[0]
    ax.plot(time, yaw, 'k', label="yaw")
    ax.set_ylabel("yaw")
    ax = axes[1]
    ax.plot(time, yaw_dot, 'k', label="yaw")
    ax.set_ylabel("yaw dot")
    ax.set_xlabel("Time, s")

    speed = np.sqrt(xdot[:,0]**2 + xdot[:,1]**2)
    (fig, axes) = plt.subplots(nrows=1, ncols=1, num="XY Trajectory")
    ax = axes
    ax.scatter(x[:,0], x[:,1], c=cm.winter(speed/speed.max()), s=4, label="Flat Output")
    ax.plot(waypoints[:,0], waypoints[:,1], 'ko', markersize=10, label="Waypoints")
    ax.grid()
    ax.legend()

    plt.show()
"""
This example will demonstrate the capability of RotorPy by evaluating an
Unscented Kalman Filter designed to do wind estimation for a quadrotor UAV. 

First, we'll import some useful Python packages.
"""

import numpy as np
import scipy as sp
import copy
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)    # For repeatability we will set the seed for the RNG

"""
Next, let's import the simulator and its important modules: a vehicle, 
controller, some trajectories, and wind. 
"""

from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.polynomial_traj import Polynomial
from rotorpy.wind.dryden_winds import DrydenGust

from rotorpy.utils.postprocessing import unpack_sim_data

"""
Last for imports we'll import the two filters we want to evaluate.
These were written prior and are found in rotorpy/estimators/
"""

from rotorpy.estimators.wind_ukf import WindUKF
from rotorpy.estimators.wind_ekf import WindEKF

"""
Next we'll define a useful function. It's a script for automatically 
fitting a quadratic drag model to the calibration data we'll collect. 
"""

def auto_system_identification(mass, body_vel, abs_acceleration, plot=False):
    """
    AUTOMATIC SYSTEM IDENTIFICATION
    Inputs:
        mass, the mass of the UAV, kg
        body_vel, the velocity of the UAV in the body axes, m/s
        abs_acceleration, absolute value of accelerometer measurements minus thrust, m/s/s
        plot, bool to plot accelerometer vs speed. 
    Outputs:
        Fitted drag coefficients for a quadratic drag model.
    """
    def parabola(x, A):
        return A*(x**2)
    
    # The main assumption is that the only other forces acting on the vehicle are drag. 
    # We assume the only source of drag is parastic drag, of the form: F_D = -c_D*|V|*V

    Ax, _ = sp.optimize.curve_fit(parabola, body_vel[:,0], abs_acceleration[:,0])
    Ay, _ = sp.optimize.curve_fit(parabola, body_vel[:,1], abs_acceleration[:,1])
    Az, _ = sp.optimize.curve_fit(parabola, body_vel[:,2], abs_acceleration[:,2])

    c_Dx = Ax*mass
    c_Dy = Ay*mass
    c_Dz = Az*mass

    if plot:
          (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num="Fitting Data")
          ax = axes[0]
          ax.plot(body_vel[:,0], abs_accel[:,0], 'r.', markersize=6, label="Data")
          ax.plot(body_vel[:,0], (c_Dx/mass)*(body_vel[:,0]**2), 'k.', markersize=6, label="Quadratic Fit")
          ax.set_ylabel("|Accel X|")
          ax = axes[1]
          ax.plot(body_vel[:,1], abs_accel[:,1], 'g.', markersize=6, label="Data")
          ax.plot(body_vel[:,1], (c_Dy/mass)*(body_vel[:,1]**2), 'k.', markersize=6, label="Quadratic Fit")
          ax.set_ylabel("|Accel Y|")
          ax = axes[2]
          ax.plot(body_vel[:,2], abs_accel[:,2], 'b.', markersize=6, label="Data")
          ax.plot(body_vel[:,2], (c_Dz/mass)*(body_vel[:,2]**2), 'k.', markersize=6, label="Quadratic Fit")
          ax.set_xlabel("Body Velocity, m/s")
          ax.set_ylabel("|Accel Z|")

    return (c_Dx[0], c_Dy[0], c_Dz[0])

"""
Next, we'll set up a calibration trajectory. The quadrotor will fly in straight lines
back and forth on each axis.
"""

dist= 5
points = np.array([[0,0,0],
                   [dist,0,0],
                   [0,0,0],
                   [0,dist,0],
                   [0,0,0],
                   [0,0,dist],
                   [0,0,0]])
calibration_traj = Polynomial(points, v_avg=1)

"""
Run the Monte Carlo simulations
"""
num_trials = 50

# Filter parameters: process noise and initial covariance
Q = 0.1*np.diag(np.concatenate([0.5*np.ones(3),
                                0.7*np.ones(3),
                                0.3*np.ones(3)
                                ]))
P0 = 1*np.eye(9)

# Arrays for analysis
mass = np.zeros((num_trials,))
cD = np.zeros((num_trials, 3))
k_d = np.zeros((num_trials,))
k_z = np.zeros((num_trials,))
wind_rmse = np.zeros((num_trials, 3))

best_wind_rmse = 10000

for i in range(num_trials):
    print("Trial %d/%d" % (i+1, num_trials))
    """
    Setup: randomly select quadrotor parameters and wind parameters.
    Instantiate the quadrotor and a stabilizing controller. 
    """
    # Randomize the parameters for the quadrotor
    trial_params = copy.deepcopy(quad_params)  # Get a copy of the quadrotor parameters (this prevents overwriting)
    trial_params['mass'] *= np.random.uniform(low=0.75, high=1.25)
    trial_params['c_Dx'] *= np.random.uniform(low=0, high=2)
    trial_params['c_Dy'] *= np.random.uniform(low=0, high=2)
    trial_params['c_Dz'] *= np.random.uniform(low=0, high=2)
    trial_params['k_d'] *= np.random.uniform(low=0, high=10)
    trial_params['k_z'] *= np.random.uniform(low=0, high=10)

    # Randomize the wind input for this trial
    wx = np.random.uniform(low=-3, high=3)
    wy = np.random.uniform(low=-3, high=3)
    wz = np.random.uniform(low=-3, high=3)
    sx = np.random.uniform(low=30, high=60)
    sy = np.random.uniform(low=30, high=60)
    sz = np.random.uniform(low=30, high=60)
    wind_profile = DrydenGust(dt=1/100, avg_wind=np.array([wx,wy,wz]), sig_wind=np.array([sx,sy,sz]))

    # Create a vehicle and controller with the ground truth parameters
    quadrotor = Multirotor(trial_params)
    se3_controller = SE3Control(trial_params)

    """
    Generate the calibration data. 
    """
    print("Generating calibration data...")
    # Initialize the quadrotor hovering. 
    hover_speed = np.sqrt(trial_params['mass']*9.81/(4*trial_params['k_eta']))
    x0 = calibration_traj.update(0)['x']
    initial_state = {'x': x0,
                     'v': np.zeros(3,),
                     'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                     'w': np.zeros(3,),
                     'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                     'rotor_speeds': np.ones((4,))*hover_speed}
    
    calibration_instance = Environment(vehicle=quadrotor,
                              controller=se3_controller,
                              trajectory=calibration_traj,
                              estimator=WindUKF(trial_params))
    calibration_instance.vehicle.initial_state = initial_state

    calibration_results = calibration_instance.run(t_final = 30,     # The duration of the environment in seconds
                                                   plot    = False   # Boolean: plots the vehicle states and commands
                                          )
    print("Done.")

    """
    Unpack the calibration data and do system identification
    """
    calibration_data = unpack_sim_data(calibration_results)

    # Extract the orientation and use it to express the velocity in the body frame
    q = calibration_data[['qx', 'qy', 'qz', 'qw']].to_numpy()
    vel = (calibration_data[['xdot', 'ydot', 'zdot']].to_numpy())[..., np.newaxis]
    R = sp.spatial.transform.Rotation.from_quat(q).as_matrix()
    RT = np.transpose(R, axes=[0,2,1]) 
    body_vel = (RT@vel)[:,:,0]

    # Extract the acceleration, compute the acceleration minus the commanded thrust to isolate aerodynamic forces
    accel = calibration_data[['ax', 'ay', 'az']].to_numpy()
    accel[:,2] -= (calibration_data['thrustdes']/trial_params['mass'])
    abs_accel = np.abs(accel)

    # Fit aero parameters
    (fit_c_Dx, fit_c_Dy, fit_c_Dz) = auto_system_identification(trial_params['mass'], body_vel, abs_accel, plot=False)
    print("Fitted aero parameters.")

    """
    Set up the estimators. There is an EKF and UKF. You can choose 
    which one to evaluate when setting up the evaluation instance of the 
    simulator. 
    """
    # Copy the trial parameters. The filtes use the parasitic drag coefficients
    # for the process model, so use the fitted coefficients from the previous step. 
    estimator_params = copy.deepcopy(trial_params)
    estimator_params['c_Dx'] = fit_c_Dx
    estimator_params['c_Dy'] = fit_c_Dy
    estimator_params['c_Dz'] = fit_c_Dz

    # Initialize the filters using the mean wind speed. 
    xhat0 = np.array([0.0, 0.0, 0.0, 0.01, 0.01, 0.01, wx, wy, wz])
    wind_ukf = WindUKF(estimator_params,Q=Q,xhat0=xhat0,P0=P0,dt=1/100,alpha=1e-3,beta=2,kappa=-1)
    wind_ekf = WindEKF(estimator_params,Q=Q,xhat0=xhat0,P0=P0,dt=1/100)

    """
    Run the evaluation. The quadrotor will hover while being subject to 
    Dryden Gust. You can switch between the UKF and EKF by changing the 
    'estimator' argument accordingly. 
    """
    print("Evaluating the filter...")
    evaluation_traj = HoverTraj()  
    evaluation_instance = Environment(vehicle=quadrotor,
                                      controller=se3_controller,
                                      trajectory=evaluation_traj,
                                      wind_profile=wind_profile,
                                      estimator=wind_ukf)

    evaluation_results = evaluation_instance.run(t_final = 10,       # The maximum duration of the environment in seconds
                                                 plot = False,
                                                 verbose= True)
    
    """
    Post process the evaluation results. 
    """
    evaluation_data = unpack_sim_data(evaluation_results)

    # Extract the orientation and use it to express the wind in the body frame
    q = evaluation_data[['qx', 'qy', 'qz', 'qw']].to_numpy()
    wind = (evaluation_data[['windx', 'windy', 'windz']].to_numpy())[..., np.newaxis]
    R = sp.spatial.transform.Rotation.from_quat(q).as_matrix()
    RT = np.transpose(R, axes=[0,2,1]) 
    body_wind = (RT@wind)[:,:,0]

    # Extract the wind states from the filter. 
    wind_est = evaluation_data[['xhat_6', 'xhat_7', 'xhat_8']].to_numpy()

    """
    Compute the RMSE between the wind estimate and the ground truth estimate. 
    """
    print("Computing and saving metrics...")
    # Compute root mean square error
    rmse = np.sqrt(((wind_est - body_wind)**2).mean(axis=0))

    avg_rmse = np.mean(rmse)
    if avg_rmse < best_wind_rmse:
         # Save this instance for plotting
         plot_body_wind = body_wind
         plot_wind_est = wind_est
         plot_time = evaluation_data['time'].to_numpy()

    """
    Save the vehicle parameters and the filter performance for plotting. 
    """

    mass[i] = trial_params['mass']
    cD[i,:] = np.array([trial_params['c_Dx'], trial_params['c_Dy'], trial_params['c_Dz']])
    k_d[i] = trial_params['k_d']
    k_z[i] = trial_params['k_z']
    wind_rmse[i,:] = rmse
    print("Done.")
    print("------------------------------------------------------------------")

"""
Plot the RMSE for each body axis. 
"""

# Box and Whisker plot
boxprops = dict(linestyle='-', linewidth=1.5, color='k')
flierprops = dict(marker='o', markerfacecolor='k', markersize=4,
                  linestyle='none')
medianprops = dict(linestyle='-', linewidth=1.5, color='r')
whiskerprops = dict(linestyle='-', linewidth=1.5, color='k')
capprops = dict(linestyle='-', linewidth=1.5, color='k')

plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.autolayout"] = True
RMSE_data = pd.DataFrame({"X Axis": wind_rmse[:,0], "Y Axis": wind_rmse[:,1], "Z Axis": wind_rmse[:,2]})
ax = RMSE_data[['X Axis', 'Y Axis', 'Z Axis']].plot(kind='box', title="", 
                                                    boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, 
                                                    whiskerprops=whiskerprops, capprops=capprops)
ax.set_ylabel("Wind RMSE, m/s")

# Plot the best performance of the filter.
linew = 1.5  # Width of the plot lines
(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num="Body Wind Comparison")
ax = axes[0]
ax.plot(plot_time, plot_body_wind[:,0], 'rx', linewidth=linew, label="Ground Truth")
ax.plot(plot_time, plot_wind_est[:,0], 'k', linewidth=linew, label="UKF Estimate")
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fancybox=True, shadow=True)
ax.set_ylabel("$w_x$, m/s")
ax = axes[1]
ax.plot(plot_time, plot_body_wind[:,1], 'gx', linewidth=linew, label="Ground Truth")
ax.plot(plot_time, plot_wind_est[:,1], 'k', linewidth=linew, label="UKF Estimate")
# ax.legend()
ax.set_ylabel("$w_y$, m/s")
ax = axes[2]
ax.plot(plot_time, plot_body_wind[:,2], 'bx', linewidth=linew, label="Ground Truth")
ax.plot(plot_time, plot_wind_est[:,2], 'k', linewidth=linew, label="UKF Estimate")
# ax.legend()
ax.set_ylabel("$w_z$, m/s")
ax.set_xlabel("Time, s")

print("Average RMSE: ", np.mean(wind_rmse,axis=0))
plt.show()
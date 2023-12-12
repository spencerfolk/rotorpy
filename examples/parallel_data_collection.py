from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.environments import Environment
from rotorpy.world import World
from rotorpy.utils.occupancy_map import OccupancyMap

import numpy as np                  # For array creation/manipulation
import matplotlib.pyplot as plt     # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import Rotation  # For doing conversions between different rotation descriptions, applying rotations, etc. 
import os                           # For path generation

import multiprocessing
import csv

from tqdm import tqdm

import time
import datetime

"""
In this script we demonstrate parallel data collection utilizing multiple CPU cores. 
The script generates random minimum snap trajectories and simulates a quadrotor tracking these trajectories. 
If the parallel flag is set to True, the script will use multiprocessing to run multiple simulations in parallel.
"""

def sample_waypoints(num_waypoints, world, world_buffer=2, check_collision=True, min_distance=1, max_distance=3, max_attempts=1000, start_waypoint=None, end_waypoint=None):
    """
    Samples random waypoints (x,y,z) in the world. Ensures waypoints do not collide with objects, although there is no guarantee that 
    the path you generate with these waypoints will be collision free. 
    Inputs:
        num_waypoints: Number of waypoints to sample. 
        world: Instance of World class containing the map extents and any obstacles. 
        world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away 
            from the edge of the world.
        check_collision: If True, checks for collisions with obstacles. If False, does not check for collisions. Checking collisions slows down the script. 
        min_distance: Minimum distance between waypoints consecutive waypoints. 
        max_distance: Maximum distance between consecutive waypoints.
        max_attempts: Maximum number of attempts to sample a waypoint.
        start_waypoint: If specified, the first waypoint will be this point. 
        end_waypoint: If specified, the last waypoint will be this point.
    Outputs:
        waypoints: A list of (x,y,z) waypoints. [[waypoint_1], [waypoint_2], ... , [waypoint_n]]
    """

    if min_distance > max_distance:
        raise Exception("min_distance must be less than or equal to max_distance.")

    def check_distance(waypoint, waypoints, min_distance, max_distance):
        """
        Checks if the waypoint is at least min_distance away from all other waypoints. 
        Inputs:
            waypoint: The waypoint to check. 
            waypoints: A list of waypoints. 
            min_distance: The minimum distance the waypoint must be from all other waypoints. 
            max_distance: The maximum distance the waypoint can be from all other waypoints.
        Outputs:
            collision: True if the waypoint is at least min_distance away from all other waypoints. False otherwise. 
        """
        collision = False
        for w in waypoints:
            if (np.linalg.norm(waypoint-w) < min_distance) or (np.linalg.norm(waypoint-w) > max_distance):
                collision = True
        return collision
    
    def check_obstacles(waypoint, occupancy_map):
        """
        Checks if the waypoint is colliding with any obstacles in the world. 
        Inputs:
            waypoint: The waypoint to check. 
            occupancy_map: An instance of the occupancy map.
        Outputs:
            collision: True if the waypoint is colliding with any obstacles in the world. False otherwise. 
        """
        collision = False
        if occupancy_map.is_occupied_metric(waypoint):
            collision = True
        return collision
    
    def single_sample(world, current_waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts=1000, rng=None):
        """
        Samples a single waypoint. 
        Inputs:
            world: Instance of World class containing the map extents and any obstacles. 
            world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away 
                from the edge of the world.
            occupancy_map: An instance of the occupancy map.
            min_distance: Minimum distance between waypoints consecutive waypoints. 
            max_distance: Maximum distance between consecutive waypoints.
            max_attempts: Maximum number of attempts to sample a waypoint.
            rng: Random number generator. If None, uses numpy's random number generator.
        Outputs:
            waypoint: A single (x,y,z) waypoint. 
        """

        num_attempts = 0

        world_lower_limits = np.array(world.world['bounds']['extents'][0::2])+world_buffer
        world_upper_limits = np.array(world.world['bounds']['extents'][1::2])-world_buffer

        if len(current_waypoints) == 0:
            max_distance_lower_limits = world_lower_limits
            max_distance_upper_limits = world_upper_limits
        else:
            max_distance_lower_limits = current_waypoints[-1] - max_distance
            max_distance_upper_limits = current_waypoints[-1] + max_distance

        lower_limits = np.max(np.vstack((world_lower_limits, max_distance_lower_limits)), axis=0)
        upper_limits = np.min(np.vstack((world_upper_limits, max_distance_upper_limits)), axis=0)

        waypoint = np.random.uniform(low=lower_limits, 
                                     high=upper_limits, 
                                     size=(3,))
        while check_obstacles(waypoint, occupancy_map) or (check_distance(waypoint, current_waypoints, min_distance, max_distance) if occupancy_map is not None else False):
            waypoint = np.random.uniform(low=lower_limits, 
                                         high=upper_limits, 
                                         size=(3,))
            num_attempts += 1
            if num_attempts > max_attempts:
                raise Exception("Could not sample a waypoint after {} attempts. Issue with obstacles: {}, Issue with min/max distance: {}".format(max_attempts, check_obstacles(waypoint, occupancy_map), check_distance(waypoint, current_waypoints, min_distance, max_distance)))
        return waypoint
    
    ######################################################################################################################

    waypoints = []

    if check_collision:
        # Create occupancy map from the world. This can potentially be slow, so only do it if the user wants to check for collisions.
        occupancy_map = OccupancyMap(world=world, resolution=[0.5, 0.5, 0.5], margin=0.1)
    else:
        occupancy_map = None

    if start_waypoint is not None: 
        waypoints = [start_waypoint]
    else:  
        # Randomly sample a start waypoint.
        waypoints.append(single_sample(world, waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts))
        
    num_waypoints -= 1

    if end_waypoint is not None:
        num_waypoints -= 1

    for _ in range(num_waypoints):
        waypoints.append(single_sample(world, waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts))

    if end_waypoint is not None:
        waypoints.append(end_waypoint)

    return np.array(waypoints)

def compute_cost(sim_result):
    """
    Computes the cost from the output of a simulator instance.
    Inputs:
        sim_result: The output of a simulator instance.
    Outputs:
        cost: The cost of the trajectory.
    """

    # Some useful values from the trajectory. 

    time = sim_result['time']
    x = sim_result['state']['x']                                    # Position
    v = sim_result['state']['v']                                    # Velocity
    q = sim_result['state']['q']                                    # Attitude
    w = sim_result['state']['w']                                    # Body rates
    rotor_speeds = sim_result['state']['rotor_speeds']              # Rotor speeds

    x_des = sim_result['flat']['x']                                 # Desired position
    v_des = sim_result['flat']['x_dot']                             # Desired velocity
    q_des = sim_result['control']['cmd_q']                          # Desired attitude
    rotor_speeds_des = sim_result['control']['cmd_motor_speeds']    # Desired rotor speeds 

    # Write your cost function here. For example this is average position error over the trajectory. 
    sim_cost = np.linalg.norm(x-x_des, axis=1).mean()

    return sim_cost

def write_to_csv(output_file, row):
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    return None

def single_minsnap_instance(world, vehicle, controller, num_waypoints, start_waypoint=None, end_waypoint=None, 
                            world_buffer=2, min_distance=1, max_distance=3, vavg=2, random_yaw=True, yaw_min=-0.85*np.pi, yaw_max=0.85*np.pi,
                            seed=None, save_trial=False):
    """
    Generate a single instance of the simulator with a minsnap trajectory. 
    Inputs:
        world: Instance of World class containing the map extents and any obstacles.
        vehicle: Instance of a vehicle class.
        controller: Instance of a controller class.
        num_waypoints: Number of waypoints to sample.
        start_waypoint: If specified, the first waypoint will be this point.
        end_waypoint: If specified, the last waypoint will be this point.
        world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away 
            from the edge of the world.
        min_distance: Minimum distance between waypoints consecutive waypoints.
        max_distance: Maximum distance between consecutive waypoints.
        vavg: Average velocity of the vehicle.
        random_yaw: If True, the yaw angles will be randomly sampled. If False, the yaw angles will be 0.
        yaw_min: The minimum yaw angle to sample.
        yaw_max: The maximum yaw angle to sample.
        seed: The seed for the random number generator. If None, uses numpy's random number generator.
        save_trial: If True, saves the trial data to a .csv file.
    Outputs:
        output: the cost of the trajectory followed by the polynomial coefficients for the position and yaw. 
    """

    if seed is not None:
        np.random.seed(seed)
        traj_id = seed
    else:
        np.random.seed()
        traj_id = np.random.randint(0, 2**16)  # Randomly assign trajectory ID

    # First sample the waypoints.
    waypoints = sample_waypoints(num_waypoints=num_waypoints, world=world, world_buffer=world_buffer, 
                                     min_distance=min_distance, max_distance=max_distance, 
                                     start_waypoint=start_waypoint, end_waypoint=end_waypoint)
    
    # Sample the yaw angles
    if random_yaw:
        yaw_angles=np.random.uniform(low=yaw_min, high=yaw_max, size=len(waypoints))
    else:
        yaw_angles=np.zeros(len(waypoints))

    # Generate the minsnap trajectory
    traj = MinSnap(points=waypoints, yaw_angles=yaw_angles, v_avg=vavg)

    # Now create an instance of the simulator and run it. 
    sim_instance = Environment(vehicle=vehicle, controller=controller, trajectory=traj, wind_profile=None, sim_rate=100)

    # Set the initial state to the first waypoint at hover. 
    x0 = {'x': waypoints[0],
          'v': np.zeros(3,),
          'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
          'w': np.zeros(3,),
          'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
          'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
    
    sim_instance.vehicle.initial_state = x0

    # Now run the simulator for the length of the trajectory. 
    sim_result = sim_instance.run(t_final = traj.t_keyframes[-1], 
                              use_mocap=False, 
                              terminate=False, 
                              plot=False, 
                              plot_mocap=False, 
                              plot_estimator=False, 
                              plot_imu=False, 
                              animate_bool=False, 
                              animate_wind=False, 
                              verbose=False)
    
    if save_trial:
        savepath = os.path.join(os.path.dirname(__file__), 'trial_data')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        sim_instance.save_to_csv(os.path.join(savepath, 'trial_{}.csv'.format(traj_id)))
    
    # Compute the cost of the trajectory from result
    trajectory_cost = compute_cost(sim_result)

    # Now extract the polynomial coefficients for the trajectory.
    pos_poly = traj.x_poly
    yaw_poly = traj.yaw_poly

    summary_output = np.concatenate((np.array([int(traj_id)]), np.array([trajectory_cost]), pos_poly.ravel(), yaw_poly.ravel()))

    return summary_output

def generate_data(output_csv_file, world, vehicle, controller, 
                  num_simulations, num_waypoints, vavg, 
                  random_yaw, yaw_min, yaw_max, 
                  world_buffer, min_distance, max_distance, 
                  start_waypoint, end_waypoint, 
                  parallel=True,
                  save_individual_trials=False):
    """
    Generates data.
    Inputs:
        output_file: The name of the output file.
        world: Instance of World class containing the map extents and any obstacles.
        vehicle: Instance of a vehicle class.
        controller: Instance of a controller class.
        num_simulations: The number of simulations to run.
        num_waypoints: The number of waypoints to sample.
        vavg: The average velocity of the vehicle.
        random_yaw: If True, the yaw angles will be randomly sampled. If False, the yaw angles will be 0.
        yaw_min: The minimum yaw angle to sample.
        yaw_max: The maximum yaw angle to sample.
        world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
            from the edge of the world.
        min_distance: Minimum distance between waypoints consecutive waypoints.
        max_distance: Maximum distance between consecutive waypoints.
        start_waypoint: If specified, the first waypoint will be this point.
        end_waypoint: If specified, the last waypoint will be this point.
    Outputs:
        None. It writes to the output file.
    """

    if not parallel:
        for _ in tqdm(range(num_simulations), desc="Running simulations (sequentially)..."):
            result = single_minsnap_instance(world, vehicle, controller, 
                                            num_waypoints, start_waypoint, end_waypoint, 
                                            world_buffer, min_distance, max_distance, vavg, 
                                            random_yaw, yaw_min, yaw_max, seed=None, save_trial=save_individual_trials)
            
            write_to_csv(output_csv_file, result)

    else:
        # Use multiprocessing to run multiple simulations in parallel.

        num_cores = min(multiprocessing.cpu_count(), 40)

        print("Running {} simulations in parallel with up to {} cores.".format(num_simulations, num_cores))
              
        pool = multiprocessing.Pool(num_cores)

        # Use numpy random to generate seeds for each simulation.
        seeds = np.random.choice(np.arange(num_simulations), size=num_simulations, replace=False)

        manager = multiprocessing.Manager()
        
        lock = manager.Lock()

        def update_results(result):
            with lock:
                write_to_csv(output_file=output_csv_file, row=result)

        code_rate = 1.33  # simulations per second, emperically determined from our machine, but will differ for yours.
        expected_duration_seconds = num_simulations/code_rate  
        
        current_time = datetime.datetime.now()
        end_time = current_time + datetime.timedelta(seconds=expected_duration_seconds)

        print(f"Start time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Expected duration: %3.2f seconds (%3.2f minutes, or %3.2f hours)" % (expected_duration_seconds, expected_duration_seconds/60, expected_duration_seconds/3600))
        print(f"Program *may* end around: {end_time.strftime('%Y-%m-%d %H:%M:%S')}, depending on your machine specs, number of waypoints, distance between waypoints, etc.")

        print("Running simulations (in parallel)...")
        for _ in range(num_simulations):
            pool.apply_async(single_minsnap_instance, args=(world, vehicle, controller, 
                                                            num_waypoints, start_waypoint, end_waypoint, 
                                                            world_buffer, min_distance, max_distance, vavg, 
                                                            random_yaw, yaw_min, yaw_max, seeds[_], save_individual_trials),
                                                            callback=update_results)
            
        pool.close()
        pool.join()

    return None

def main(num_simulations, parallel_bool, save_trials=False):
    """
    Main function for generating data.
    Inputs:
        num_simulations: The number of simulations to run.
        parallel_bool: If True, runs the simulations in parallel. If False, runs the simulations sequentially.
        save_trials: If True, saves each trial data to a separate .csv file. Uses more memory, but allows you to see the results of each trial at a later date.
    Outputs:
        None. It writes to the output file.
    """

    world_size = 10
    num_waypoints = 4
    vavg = 2
    random_yaw = False
    yaw_min = -0.85*np.pi
    yaw_max = 0.85*np.pi

    world_buffer = 2
    min_distance = 1
    max_distance = min_distance+3
    start_waypoint = None               # If you want to start at a specific waypoint, specify it using [xstart, ystart, zstart]
    end_waypoint = None                 # If you want to end at a specific waypoint, specify it using [xend, yend, zend]

    # Create the output file
    output_csv_file = os.path.dirname(__file__) + '/data.csv'

    if os.path.exists(output_csv_file):
        # Ask the user if they want to remove the existing file.
        user_input = input("The file {} already exists. Do you want to remove the existing file? (y/n)".format(output_csv_file))
        if user_input == 'y':
            # Remove the existing file
            os.remove(output_csv_file)
        elif user_input == 'n':
            raise Exception("Please delete or rename the file {} before running this script.".format(output_csv_file))
        else:
            raise Exception("Invalid input. Please enter 'y' or 'n'.")

    if save_trials:
        savepath = os.path.join(os.path.dirname(__file__), 'trial_data')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        else:
            # Ask the user if they want to remove the existing files in the directory.
            user_input = input("The directory {} already exists. Do you want to remove the existing files? (y/n)".format(savepath))
            if user_input == 'y':
                # Remove existing files in the directory
                for file in os.listdir(savepath):
                    os.remove(os.path.join(savepath, file))
            elif user_input == 'n':
                raise Exception("Please delete or rename the files in the directory {} before running this script.".format(savepath))
            else:
                raise Exception("Invalid input. Please enter 'y' or 'n'.")

    # Append headers to the output file
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # This depends on the number of waypoints and the order of the polynomial. Currently pos is 7th order and yaw is 7th order.
        writer.writerow(['traj_number'] + ['cost'] 
                + ['x_poly_seg_{}_coeff_{}'.format(i,j) for i in range(num_waypoints-1) for j in range(8)]
                + ['y_poly_seg_{}_coeff_{}'.format(i,j) for i in range(num_waypoints-1) for j in range(8)]
                + ['z_poly_seg_{}_coeff_{}'.format(i,j) for i in range(num_waypoints-1) for j in range(8)]
                + ['yaw_poly_seg_{}_coeff_{}'.format(i,j) for i in range(num_waypoints-1) for j in range(8)])  

    # Now create the world, vehicle, and controller objects.
    world = World.empty([-world_size/2, world_size/2, -world_size/2, world_size/2, -world_size/2, world_size/2])
    vehicle = Multirotor(quad_params)
    controller = SE3Control(quad_params)

    # Generate the data
    start_time = time.time()
    generate_data(output_csv_file, world, vehicle, controller,
                  num_simulations, num_waypoints, vavg, 
                  random_yaw, yaw_min, yaw_max, 
                  world_buffer, min_distance, max_distance, 
                  start_waypoint, end_waypoint,
                  parallel=parallel_bool,
                  save_individual_trials=save_trials)
    end_time = time.time()
    print("Time elapsed: %3.2f seconds, parallel: %s" % (end_time-start_time, parallel_bool))

    return None

if __name__=="__main__":

    main(num_simulations=10, parallel_bool=True, save_trials=True)
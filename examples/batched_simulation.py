import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from rotorpy.trajectories.minsnap import MinSnap, BatchedMinSnap
from rotorpy.vehicles.multirotor import Multirotor, BatchedMultirotorParams, BatchedMultirotor
from rotorpy.controllers.quadrotor_control import BatchedSE3Control, SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params as cf_quad_params
from rotorpy.vehicles.hummingbird_params import quad_params as hb_quad_params
from rotorpy.utils.trajgen_utils import generate_random_minsnap_traj
from rotorpy.world import World
from rotorpy.wind.default_winds import NoWind, BatchedNoWind
from rotorpy.simulate import simulate, simulate_batch
from rotorpy.sensors.imu import Imu, BatchedImu
from rotorpy.sensors.external_mocap import MotionCapture
from rotorpy.estimators.nullestimator import NullEstimator


def main():
    # This prevents pytorch from spawning multiple threads. Commenting this line out can improve performance
    # on CPU for larger batch sizes. But can also cause issues if you want to do your own multiprocessing, 
    # e.g. with the multiprocessing python module.
    torch.set_num_threads(1)

    # set based on if you want to use GPU or CPU for simulation. CPU is faster for smaller batch sizes.
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    # How many drones to simulate in parallel. Up to a limit, increasing this value increases the efficiency (average FPS)
    # of the simulation. How many you can simulate in parallel efficiently will depend on your machine.
    num_drones = 50

    #### Initial Drone States ####
    # We create initial states for each drone in the batch.
    # In this case, we're setting the initial state of every drone to be basically 0.
    # Note that we're going to be working in torch (as opposed to numpy in standard RotorPy), and using the double type.
    init_rotor_speed = 1788.53
    x0 = {'x': torch.zeros(num_drones,3, device=device).double(),
          'v': torch.zeros(num_drones, 3, device=device).double(),
          'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_drones, 1).double(),
          'w': torch.zeros(num_drones, 3, device=device).double(),
          'wind': torch.zeros(num_drones, 3, device=device).double(),
          'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_drones, 1).double()}


    #### Generate Trajectories ####
    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    num_waypoints = 4
    v_avg_des = 2.0

    # when interfacing with some of the standard rotorpy hardware, you'll have to convert from torch -> numpy.
    positions = x0['x'].cpu().numpy()
    trajectories = []
    ref_traj_gen_start_time = time.time()

    # Generate the same trajectories each time.
    np.random.seed(10)
    num_done = 0
    while num_done < num_drones:
        traj = generate_random_minsnap_traj(world, num_waypoints, v_avg_des, min_distance=1.0, max_distance=2.0,
                                            start_position=positions[num_done])
        if traj is not None:
            trajectories.append(traj)
            num_done += 1

    # Set to 0 if you want sim results to be more deterministic (default value is 100)
    cf_quad_params["motor_noise_std"] = 0
    hb_quad_params["motor_noise_std"] = 0

    # control_abstraction = "cmd_motor_speeds"  # the default abstraction
    control_abstraction = "cmd_ctatt"

    # We'll simulate half crazyflies, half hummingbirds
    all_quad_params = [cf_quad_params]*(num_drones//2) + [hb_quad_params]*(num_drones//2)

    # Optional: specify feedback gains for each drone in the batch. (can be different for each drone)
    kp_pos = torch.tensor([6.5, 6.5, 15]).repeat(num_drones, 1).double()
    kd_pos = torch.tensor([4.0, 4.0, 9]).repeat(num_drones, 1).double()
    kp_att = torch.tensor([544]).repeat(num_drones, 1).double()
    kd_att = torch.tensor([46.64]).repeat(num_drones, 1).double()

    # Collate all the individual MinSnap objects into a single BatchedMinSnap object, which allows us to compute
    # reference commands for all trajectories at the same time.
    batched_trajs = BatchedMinSnap(trajectories, device=device)
    print(f"Time to Generate reference trajectories: {time.time() - ref_traj_gen_start_time}")

    # Define this object which contains dynamics params for each of the drones.
    # If the batch size is large, this can save memory by sharing the dynamics params across the controller and
    # multirotor object.
    batch_params = BatchedMultirotorParams(all_quad_params, num_drones, device)

    # Define a batched controller object which lets us compute control inputs for all drones in the batch at the
    # same time. Note that currently, all drones in the batch must share the same quad_params.
    controller = BatchedSE3Control(batch_params, num_drones, device=device,
                                   kp_pos=kp_pos,
                                   kd_pos=kd_pos,
                                   kp_att=kp_att,
                                   kd_att=kd_att)

    # Define a batched multirotor, which simulates all drones in the batch simultaneously.
    # Choose 'dopri5' to mimic scipy's default solve_ivp behavior with an adaptive step size, or 'rk4'
    # for a fixed step-size integrator, which is lower-fidelity but much faster.
    vehicle = BatchedMultirotor(batch_params, num_drones, x0, device=device, integrator='dopri5', control_abstraction=control_abstraction)

    dt = 0.01

    # Optional: define when each drone in the batch should terminate.
    t_fs = np.array([trajectory.t_keyframes[-1] for trajectory in trajectories])

    # Define a wind profile -- for batched drones, only NoWind and ConstantWind are supported rn.
    wind_profile = BatchedNoWind(num_drones)

    # Define a BatchedIMU object, which simulates noisy IMU measurements
    batched_imu = BatchedImu(num_drones, device=device)

    # Call the simulate_batch function, which will simulate all drones using the vectorized dynamics.
    sim_fn_start_time = time.time()
    results = simulate_batch(world, x0, vehicle, controller, batched_trajs, wind_profile, batched_imu, t_fs, dt, 0.25, print_fps=False)
    sim_fn_end_time = time.time()
    print(f"time to simulate {num_drones} batched using simulate_batch() fn: {sim_fn_end_time - sim_fn_start_time}")

    # A dict containing arrays of shape (N, num_drones, ...), where N is the number of timesteps it took for the
    # last drone to terminate. Has the same keys as the state dict returned by the standard simulate() function.
    simulate_fn_states = results[1]

    # Contains the timesteps at which each drone terminated.
    simulate_fn_done_times = results[-1]
    print(f"FPS of batched simulation was {np.sum(simulate_fn_done_times)/(sim_fn_end_time - sim_fn_start_time)}")

    # Contains the exit statuses for each drone.
    exit_statuses = results[-2]

    #### Sequential Simulation ####
    # For comparison, we'll also simulate a standard Multirotor.
    x0_single = {'x': np.array([0, 0, 0]),
                 'v': np.zeros(3, ),
                 'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
                 'w': np.zeros(3, ),
                 'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
                 'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

    all_seq_states = []
    mocap_params = {'pos_noise_density': 0.0005*np.ones((3,)),  # noise density for position
                    'vel_noise_density': 0.0010*np.ones((3,)),          # noise density for velocity
                    'att_noise_density': 0.0005*np.ones((3,)),          # noise density for attitude
                    'rate_noise_density': 0.0005*np.ones((3,)),         # noise density for body rates
                    'vel_artifact_max': 5,                              # maximum magnitude of the artifact in velocity (m/s)
                    'vel_artifact_prob': 0.001,                         # probability that an artifact will occur for a given velocity measurement
                    'rate_artifact_max': 1,                             # maximum magnitude of the artifact in body rates (rad/s)
                    'rate_artifact_prob': 0.0002                        # probability that an artifact will occur for a given rate measurement
                    }
    mocap = MotionCapture(sampling_rate=int(1/dt), mocap_params=mocap_params, with_artifacts=False)

    series_start_time = time.time()
    total_time = 0
    total_frames = 0
    for d in range(num_drones):
        controller_single = SE3Control(all_quad_params[d])
        vehicle_single = Multirotor(all_quad_params[d], initial_state=x0_single, control_abstraction=control_abstraction)
        start_time = time.time()
        single_result = simulate(world, x0_single, vehicle_single, controller_single, trajectories[d],
                                 NoWind(), Imu(sampling_rate=int(1/dt)), mocap, NullEstimator(),
                                 trajectories[d].t_keyframes[-1], dt, 0.25, use_mocap=False, print_fps=False)

        all_seq_states.append(single_result[1])
        total_frames += len(single_result[0])
        total_time += time.time() - start_time
    print(f"time to simulate {num_drones} sequentially: {time.time() - series_start_time}")
    print(f"average fps of series simulation was {total_frames/total_time}")

    ### Comparisons ###
    # Plot positions and body rates of 3 random drones and compare the batched/sequential simulations.
    num_to_plot = 3
    fig, ax = plt.subplots(num_to_plot, 3)
    which_sim = np.random.choice(num_drones, num_to_plot, replace=False)
    dims = ["x (m)", "y (m)", "z (m)"]
    for j, sim_idx in enumerate(which_sim):
        ts = np.arange(trajectories[sim_idx].t_keyframes[-1]+dt, step=dt)
        print(
            f"All values of state after tstep {simulate_fn_done_times[sim_idx]} for drone {sim_idx} are NaN: {np.all(np.isnan(simulate_fn_states['x'][simulate_fn_done_times[sim_idx]:, int(sim_idx)]))}")
        for dimension in range(3):
            ax[j][dimension].plot(ts, [trajectories[sim_idx].update(t)['x'][dimension] for t in ts], label='reference')
            ax[j][dimension].plot(ts, all_seq_states[int(sim_idx)]['x'][:,dimension], label='sequential')
            # Note that we are using ":simulate_fn_done_times[sim_idx]" to get only the states for this drone.
            # for timesteps > simulate_fn_done_times[sim_idx], the states will be nan for this drone.
            ax[j][dimension].plot(ts, simulate_fn_states['x'][:simulate_fn_done_times[sim_idx], int(sim_idx), dimension], label='batched')
            ax[j][dimension].legend()
            ax[j][dimension].set_ylabel(dims[dimension])
            ax[j][dimension].set_xlabel("Time (s)")
    fig.tight_layout()

    # Plot Body Rates
    fig2, ax2 = plt.subplots(num_to_plot, 3)
    dims = ["w_x", "w_y", "w_z"]
    for j, sim_idx in enumerate(which_sim):
        ts = np.arange(trajectories[sim_idx].t_keyframes[-1]+dt, step=dt)
        for dimension in range(3):
            ax2[j][dimension].plot(ts, all_seq_states[int(sim_idx)]['w'][:,dimension], label='sequential')
            ax2[j][dimension].plot(ts, simulate_fn_states['w'][:simulate_fn_done_times[sim_idx], int(sim_idx), dimension], label='batched')
            ax2[j][dimension].legend()
            ax2[j][dimension].set_ylabel(dims[dimension])
            ax[j][dimension].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

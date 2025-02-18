import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from rotorpy.trajectories.minsnap import MinSnap, BatchedMinSnap
from rotorpy.vehicles.batched_multirotor import BatchedMultirotor, merge_rotorpy_states, merge_flat_outputs
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3ControlBatch, SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.utils.trajgen_utils import sample_waypoints
from rotorpy.world import World
from rotorpy.wind.default_winds import NoWind
from rotorpy.batch_simulate import simulate_batch
from rotorpy.simulate import simulate
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture
from rotorpy.estimators.nullestimator import NullEstimator


def batched_worker(trajectory, vehicle, controller, x0):
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy('file_system')
    t = 0
    dt = 0.01
    t_f = 3
    try:
        flats = [trajectory.update(t)]
        states = [x0]
        controls = [controller.update(states[-1], flats[-1])]
    except Exception as e:
        raise e

    while t < t_f:
        t += dt
        states.append(vehicle.step(states[-1], controls[-1], dt))
        flats.append(trajectory.update(t))
        controls.append(controller.update(states[-1], flats[-1]))
    return states, flats, controls


def test_simulate_fn():
    print("testing batched simulation function")
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cpu")
    #### Initial Drone States ####
    num_drones = 5
    init_rotor_speed = 1788.53
    x0 = {'x': torch.zeros(num_drones,3).double(),
          'v': torch.zeros(num_drones, 3).double(),
          'q': torch.tensor([0, 0, 0, 1]).repeat(num_drones, 1).double(), # [i,j,k,w]
          'w': torch.zeros(num_drones, 3).double(),
          'wind': torch.zeros(num_drones, 3).double(),  # Since wind is handled elsewhere, this value is overwritten
          'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed]).repeat(num_drones, 1).double()}


    #### Generate Trajectories ####
    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    num_waypoints = 3
    v_avg_des = 2.0
    positions = x0['x']
    trajectories = []
    ref_traj_gen_start_time = time.time()
    num_done = 0
    while num_done < num_drones:
        waypoints = np.array(sample_waypoints(num_waypoints, world, start_waypoint=positions[num_done].cpu().numpy(),
                                              min_distance=1.0, max_distance=2.0))
        try:
            traj = MinSnap(waypoints, v_avg=v_avg_des, verbose=False)
            if traj.x_poly is not None and traj.yaw_poly is not None and traj.x_poly.shape==(num_waypoints-1,3,8):
                trajectories.append(traj)
                num_done += 1
        except TypeError:
            continue

    batched_trajs = BatchedMinSnap(trajectories, device=device)
    controller = SE3ControlBatch(quad_params, device=device)
    vehicle = BatchedMultirotor(quad_params, num_drones, x0, device=device)

    t_fs = np.array([trajectory.t_keyframes[-1] for trajectory in trajectories])
    print(f"Time to Generate reference trajectories: {time.time() - ref_traj_gen_start_time}")
    wind_profile = NoWind(num_drones)
    t_step = 0.01
    results = simulate_batch(world, x0, vehicle, controller, batched_trajs, wind_profile, t_fs, 0.01, 0.25)
    done_times = results[-1]
    states = results[1]

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(1)
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    #### Initial Drone States ####
    num_drones = 100
    init_rotor_speed = 1788.53
    x0 = {'x': torch.zeros(num_drones,3, device=device).double(),
          'v': torch.zeros(num_drones, 3, device=device).double(),
          'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_drones, 1).double(), # [i,j,k,w]
          'w': torch.zeros(num_drones, 3, device=device).double(),
          'wind': torch.zeros(num_drones, 3, device=device).double(),  # Since wind is handled elsewhere, this value is overwritten
          'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_drones, 1).double()}

    x0_single = {'x': np.array([0, 0, 0]),
     'v': np.zeros(3, ),
     'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
     'w': np.zeros(3, ),
     'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
     'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

     #### Generate Trajectories ####
    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    num_waypoints = 4
    v_avg_des = 2.0
    positions = x0['x']
    trajectories = []
    ref_traj_gen_start_time = time.time()
    num_done = 0
    np.random.seed(10)   # deterministic for testing purposes
    while num_done < num_drones:
        waypoints = np.array(sample_waypoints(num_waypoints, world, start_waypoint=positions[num_done].cpu().numpy(),
                                              min_distance=1.0, max_distance=2.0))
        try:
            traj = MinSnap(waypoints, v_avg=v_avg_des, verbose=False)
            if traj.x_poly is not None and traj.yaw_poly is not None and traj.x_poly.shape==(num_waypoints-1,3,8):
                trajectories.append(traj)
                num_done += 1
        except TypeError:
            continue

    kp_pos = torch.tensor([6.5, 6.5, 15]).repeat(num_drones, 1).double()
    kd_pos = torch.tensor([4.0, 4.0, 9]).repeat(num_drones, 1).double()
    kp_att = torch.tensor([544]).repeat(num_drones, 1).double()
    kd_att = torch.tensor([46.64]).repeat(num_drones, 1).double()

    batched_trajs = BatchedMinSnap(trajectories, device=device)
    controller = SE3ControlBatch(quad_params, num_drones, device=device,
                                 kp_pos=kp_pos,
                                 kd_pos=kd_pos,
                                 kp_att=kp_att,
                                 kd_att=kd_att)
    vehicle = BatchedMultirotor(quad_params, num_drones, x0, device=device)
    print(f"Time to Generate reference trajectories: {time.time() - ref_traj_gen_start_time}")

    controller_single = SE3Control(quad_params)
    vehicle_single = Multirotor(quad_params, initial_state=x0_single)
    vehicle_single.motor_noise = 0

    dt = 0.01

    # mp_batched_start_time = time.time()
    # num_mp_sims = 10
    # args = [(BatchedMinSnap(trajectories, device=device), BatchedMultirotor(quad_params, num_drones, dict(x0), device=device), SE3ControlBatch(quad_params, device=device), dict(x0)) for _ in range(num_mp_sims)]
    # pool = mp.Pool(processes=num_mp_sims)
    # results = pool.starmap(batched_worker, args)
    # print(f"time to simulate {num_drones * num_mp_sims} using mp and batching: {time.time() - mp_batched_start_time}")

    sim_fn_start_time = time.time()
    t_fs = np.array([trajectory.t_keyframes[-1] for trajectory in trajectories])

    wind_profile = NoWind(num_drones)
    results = simulate_batch(world, x0, vehicle, controller, batched_trajs, wind_profile, t_fs, dt, 0.25, print_fps=False)
    simulate_fn_states = results[1]
    simulate_fn_done_times = results[-1]
    exit_statuses = results[-2]
    print(f"time to simulate {num_drones} batched using simulate() fn infra : {time.time() - sim_fn_start_time}")

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
        start_time = time.time()
        single_result = simulate(world, x0_single, vehicle_single, controller_single, trajectories[d],
                                 NoWind(), Imu(sampling_rate=int(1/dt)), mocap, NullEstimator(),
                                 trajectories[d].t_keyframes[-1], dt, 0.25, use_mocap=False, print_fps=False)

        all_seq_states.append(single_result[1])
        total_frames += len(single_result[0])
        total_time += time.time() - start_time
    print(f"time to simulate {num_drones} sequentially: {time.time() - series_start_time}")
    print(f"average fps of series simulation was {total_frames/total_time}")

    num_to_plot = 3
    fig, ax = plt.subplots(num_to_plot, 3)
    which_sim = np.random.choice(num_drones, num_to_plot, replace=False)
    dims = ["x", "y", "z"]
    for j, sim_idx in enumerate(which_sim):
        for dimension in range(3):
            ax[j][dimension].plot([trajectories[sim_idx].update(t)['x'][dimension] for t in np.arange(trajectories[sim_idx].t_keyframes[-1], step=dt)], label='reference')
            ax[j][dimension].plot(all_seq_states[int(sim_idx)]['x'][:,dimension], label='sequential')
            ax[j][dimension].plot(simulate_fn_states['x'][:simulate_fn_done_times[sim_idx], int(sim_idx), dimension], label='batched')
            ax[j][dimension].legend()
            ax[j][dimension].set_ylabel(dims[dimension])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

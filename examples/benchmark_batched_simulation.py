import numpy as np
import torch
import resource
import warnings

import time
import matplotlib.pyplot as plt
import copy

from rotorpy.trajectories.minsnap import MinSnap, BatchedMinSnap
from rotorpy.vehicles.batched_multirotor import BatchedMultirotor, BatchedDynamicsParams
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import BatchedSE3Control, SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.utils.trajgen_utils import sample_waypoints
from rotorpy.world import World
from rotorpy.wind.default_winds import NoWind
from rotorpy.batch_simulate import simulate_batch
from rotorpy.simulate import simulate
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture
from rotorpy.estimators.nullestimator import NullEstimator

INIT_ROTOR_SPEED = 1788.53


def get_batch_initial_states(num_drones, device):
    x0 = {'x': torch.zeros(num_drones,3, device=device).double(),
          'v': torch.zeros(num_drones, 3, device=device).double(),
          'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_drones, 1).double(), # [i,j,k,w]
          'w': torch.zeros(num_drones, 3, device=device).double(),
          'wind': torch.zeros(num_drones, 3, device=device).double(),  # Since wind is handled elsewhere, this value is overwritten
          'rotor_speeds': torch.tensor([INIT_ROTOR_SPEED]*4, device=device).repeat(num_drones, 1).double()}
    return x0


def get_single_initial_state():
    x0_single = {'x': np.array([0, 0, 0]),
                 'v': np.zeros(3, ),
                 'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
                 'w': np.zeros(3, ),
                 'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
                 'rotor_speeds': np.array([INIT_ROTOR_SPEED] * 4)}
    return x0_single


def run_sequential_sim(traj, world, x0, dt):
    controller_single = SE3Control(quad_params)
    vehicle_single = Multirotor(quad_params, initial_state=x0)
    # vehicle_single.motor_noise = 0
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

    total_time = 0
    total_frames = 0
    start_time = time.time()
    single_result = simulate(world, x0, vehicle_single, controller_single, traj,
                             NoWind(), Imu(sampling_rate=int(1/dt)), mocap, NullEstimator(),
                             traj.t_keyframes[-1], dt, 0.25, use_mocap=False, print_fps=False)

    total_frames += len(single_result[0])
    total_time += time.time() - start_time
    return single_result[1], total_frames/total_time


# NOTE(hersh500): in this test, every drone in the batch follows the same reference trajectory. This means that all drones should
# finish at the same time. In practice, if you have a different trajectory for each drone, some will finish earlier than others,
# which will change the actual average FPS you obtain with the batched simulation (will probably make it worse).

# Example results on a machine with AMD Ryzen 9 3900X, 32GB RAM, NVidia 2080 Super, using 'dopri5' integration:
# (Measuring GPU VRAM usage is imprecise and was obtained through monitoring nvidia-smi)
# seq fps was 600.5703604578299
# Peak memory usage so far = 462.84375 Mb
# For batch size 2, CPU FPS = 24.122956586907613
# Peak memory usage so far = 465.140625 Mb
# For batch size 10, CPU FPS = 125.15730114973458
# Peak memory usage so far = 468.60546875 Mb
# For batch size 20, CPU FPS = 246.31885898407677
# Peak memory usage so far = 476.85546875 Mb
# For batch size 50, CPU FPS = 583.9553234830923
# Peak memory usage so far = 492.578125 Mb
# For batch size 100, CPU FPS = 1160.1165827787581
# Peak memory usage so far = 744.48046875 Mb
# For batch size 1000, CPU FPS = 6638.209940893531
# Peak memory usage so far = 1947.76953125 Mb
# For batch size 5000, CPU FPS = 9338.591548148928
# Peak memory usage so far = 3770.22265625 Mb
# For batch size 10000, CPU FPS = 9622.379744860333
# Peak memory usage so far = 7002.85546875 Mb
# For batch size 20000, CPU FPS = 9733.3694836387

# batch size 2 VRAM used: 204 Mb
# For batch size 2, GPU FPS = 11.774601581262745
# batch size 10 VRAM used: 204 Mb
# For batch size 10, GPU FPS = 58.96754820691534
# batch size 20 VRAM used: 204 Mb
# For batch size 20, GPU FPS = 118.45858057907071
# batch size 50 VRAM used: 210 Mb
# For batch size 50, GPU FPS = 291.9745161760285
# batch size 100 VRAM used: 216 Mb
# For batch size 100, GPU FPS = 573.5845244153943
# batch size 1000 VRAM used: 336 Mb
# For batch size 1000, GPU FPS = 5490.737933313098
# batch size 5000 VRAM used: 884 Mb
# For batch size 5000, GPU FPS = 22342.409595685356
# batch size 10000 VRAM used: 1490 Mb
# For batch size 10000, GPU FPS = 31395.2664199141
# batch size 20000 VRAM used: 2758 Mb
# For batch size 20000, GPU FPS = 39939.907530636054
def main():
    # Performance for larger (>5000) batch sizes degrades on CPU when you do this (and don't use multiprocessing),
    # But this step is necessary whenever using batched simulation with CPU multiprocessing.
    torch.set_num_threads(1)

    batch_sizes = [2, 10, 20, 50, 100, 1000, 5000, 10000, 20000]
    cpu_fps = []
    gpu_fps = []
    cpu_times = []
    gpu_times = []

    cpu_ram_usage = []

    x0 = get_single_initial_state()
    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    num_waypoints = 4
    v_avg_des = 2.0
    sim_dt = 0.01
    position = x0['x']
    np.random.seed(100)  # generate the same trajectory
    waypoints = np.array(sample_waypoints(num_waypoints, world, start_waypoint=position,
                                          min_distance=1.0, max_distance=2.0))
    traj = MinSnap(waypoints, v_avg=v_avg_des, verbose=False)

    seq_states, seq_fps = run_sequential_sim(traj, world, x0, sim_dt)
    seq_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    cpu_ram_usage.append(seq_ram_usage)
    warnings.warn("On MacOS, since system memory usage is reported differently, the reported RAM usage will be 1000x higher than actual.")
    print(f"peak memory usage for sequential simulation was {seq_ram_usage}")
    print(f"seq fps was {seq_fps}")

    device = torch.device("cpu")
    # integrator = 'dopri5'   # 'rk4' for fixed step size (faster)
    integrator = "rk4"
    # Get CPU FPS
    for batch_size in batch_sizes:
        all_quad_params = [dict(quad_params) for _ in range(batch_size)]
        batch_params = BatchedDynamicsParams(all_quad_params, batch_size, device)
        trajectories = [copy.deepcopy(traj) for _ in range(batch_size)]  # keep trajectory constant to eliminate one variable
        initial_states = get_batch_initial_states(batch_size, device)
        controller = BatchedSE3Control(batch_params, batch_size, device=device)
        vehicle = BatchedMultirotor(batch_params, batch_size, initial_states, device=device, integrator=integrator)
        t_fs = np.array([trajectory.t_keyframes[-1] for trajectory in trajectories])
        batched_traj = BatchedMinSnap(trajectories, device=device)
        wind_profile = NoWind(batch_size)

        start_time = time.time()
        results = simulate_batch(world, initial_states, vehicle, controller, batched_traj, wind_profile, t_fs, sim_dt, 0.25, print_fps=False)
        cpu_ram_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        print(f"Peak memory usage so far = {cpu_ram_usage[-1]} mb")
        total_frames = np.sum(results[-1])  # sum the timesteps at which each drone finished.
        done_time = time.time() - start_time
        cpu_fps.append(total_frames/done_time)
        cpu_times.append(done_time)
        print(f"For batch size {batch_size}, CPU FPS = {total_frames/done_time}")


    # get GPU FPS
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        for batch_size in batch_sizes:
            all_quad_params = [dict(quad_params) for _ in range(batch_size)]
            batch_params = BatchedDynamicsParams(all_quad_params, batch_size, device)
            trajectories = [copy.deepcopy(traj) for _ in range(batch_size)]  # keep trajectory constant to eliminate one variable
            initial_states = get_batch_initial_states(batch_size, device)
            controller = BatchedSE3Control(batch_params, batch_size, device=device)
            vehicle = BatchedMultirotor(batch_params, batch_size, initial_states, device=device, integrator=integrator)
            t_fs = np.array([trajectory.t_keyframes[-1] for trajectory in trajectories])
            batched_traj = BatchedMinSnap(trajectories, device=device)
            wind_profile = NoWind(batch_size)

            start_time = time.time()
            results = simulate_batch(world, initial_states, vehicle, controller, batched_traj, wind_profile, t_fs, sim_dt, 0.25, print_fps=False)
            total_frames = np.sum(results[-1])  # sum the timesteps at which each drone finished.
            done_time = time.time() - start_time
            gpu_fps.append(total_frames/done_time)
            gpu_times.append(done_time)
            print(f"For batch size {batch_size}, GPU FPS = {total_frames/done_time}")

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].plot(batch_sizes, cpu_fps, label="CPU")
    if len(gpu_fps) > 0:
        ax[0].plot(batch_sizes, gpu_fps, label="GPU")
    ax[0].axhline(seq_fps, linestyle="--", label="Sequential")
    ax[0].set_title(f"Batch Size vs. Obtained FPS with integrator {integrator}")
    ax[0].set_xlabel("Batch Size")
    ax[0].set_ylabel("FPS")
    ax[0].legend()

    ax[1].plot(batch_sizes, cpu_times, label="CPU")
    if len(gpu_times) > 0:
        ax[1].plot(batch_sizes, gpu_times, label="GPU")
    ax[1].set_title(f"Batch Size vs. Wall-Clock Time with integrator {integrator}")
    ax[1].set_xlabel("Batch Size")
    ax[1].set_ylabel("Time Taken")
    ax[1].legend()

    ax[2].plot([1] + batch_sizes, cpu_ram_usage)
    ax[2].set_title("Batch Size vs. CPU RAM Usage")
    ax[2].set_ylabel("RAM used (MB)")
    ax[2].set_xlabel("Batch Size")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
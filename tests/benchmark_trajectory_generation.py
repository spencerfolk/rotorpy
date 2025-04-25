import matplotlib.pyplot as plt
import numpy as np
import time
import os

from rotorpy.trajectories.circular_traj import BatchedThreeDCircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous, BatchedTwoDLissajous
from rotorpy.trajectories.batched_traj import BatchedTrajectory

def generate_circle_params(M):
    centers = np.random.uniform(-10, 10, (M, 3))
    radii = np.random.uniform(1, 5, (M, 3))
    freqs = np.random.uniform(0.2, 0.5, (M, 3))
    yaw_bools = np.random.choice([0, 1], M)
    return centers, radii, freqs, yaw_bools

def generate_lissajous_params(M):
    As = np.random.uniform(1, 5, M)
    Bs = np.random.uniform(1, 5, M)
    a_s = np.random.uniform(0.2, 0.5, M)
    b_s = np.random.uniform(0.2, 0.5, M)
    deltas = np.random.uniform(0, 2*np.pi, M)
    heights = np.random.uniform(0, 5, M)
    x_offsets = np.random.uniform(-10, 10, M)
    y_offsets = np.random.uniform(-10, 10, M)
    yaw_bools = np.random.choice([0, 1], M)
    return As, Bs, a_s, b_s, deltas, x_offsets, y_offsets, heights, yaw_bools

def benchmark_batched_traj(Ms, sequential_traj_obj, batched_traj_obj, param_generator_fn, traj_type=""):
    """
    Test the performance of 
    """

    def time_traj(M, traj_obj, t):

        x = np.zeros((M, 3, t.size))
        start = time.time()
        for i in range(t.size):
            flat_output = traj_obj.update(t[i])
            x[:,:,i] = flat_output['x']
        end = time.time()
        return x, end-start

    t = np.linspace(0, 10, 100)
    times_batched = []
    times_sequential = []

    print("-------------- TRAJ BENCHMARK --------------")
    print("Trajectory type: ", traj_type)
    for M in Ms:
        # Generate batched traj 
        params = param_generator_fn(M)
        batched_traj = batched_traj_obj(*params)
        single_traj_list = [sequential_traj_obj(*param) for param in zip(*params)]
        sequential_traj = BatchedTrajectory(single_traj_list)
        
        x_batched, time_batched = time_traj(M, batched_traj, t)
        times_batched.append(time_batched)

        x_individual, time_sequential = time_traj(M, sequential_traj, t)
        times_sequential.append(time_sequential)

        assert np.allclose(x_batched, x_individual), "Batched and individual trajectories do not match"

        print("For M = {}, batched time = {}, sequential time = {}".format(M, time_batched, time_sequential))
    print("--------------------------------------------")

    fig, ax = plt.subplots()
    ax.plot(Ms, times_batched, label='Batched')
    ax.plot(Ms, times_sequential, label='Sequential')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Number of UAVs')
    ax.set_ylabel('Time (s)')
    ax.set_title('Batched vs Individual Trajectory Execution Time - '+traj_type)
    ax.legend()

    fig.savefig(os.path.join(os.path.dirname(__file__), traj_type+'batched_traj_benchmark.png'))
    plt.close(fig)

if __name__=="__main__":

    Ms = [1, 10, 50, 100, 500, 1000, 5000, 10000]

    ### Circular Trajectory Benchmark
    benchmark_batched_traj(Ms, ThreeDCircularTraj, BatchedThreeDCircularTraj, generate_circle_params, traj_type="Circular")

    ### Lissajous Trajectory Benchmark
    benchmark_batched_traj(Ms, TwoDLissajous, BatchedTwoDLissajous, generate_lissajous_params, traj_type="Lissajous")
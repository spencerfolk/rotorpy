import matplotlib.pyplot as plt
import numpy as np
import time
import os

def benchmark_batched_circular_traj():
    """
    Test the performance of 
    """

    from rotorpy.trajectories.circular_traj import BatchedThreeDCircularTraj, ThreeDCircularTraj
    from rotorpy.trajectories.batched_traj import BatchedTrajectory

    def generate_circle_params(M):
        centers = np.random.uniform(-10, 10, (M, 3))
        radii = np.random.uniform(1, 5, (M, 3))
        freqs = np.random.uniform(0.2, 0.5, (M, 3))
        yaw_bools = np.random.choice([0, 1], M)
        return centers, radii, freqs, yaw_bools

    def time_traj(M, traj_obj, t):

        x = np.zeros((M, 3, t.size))
        start = time.time()
        for i in range(t.size):
            flat_output = traj_obj.update(t[i])
            x[:,:,i] = flat_output['x']
        end = time.time()
        return x, end-start

    t = np.linspace(0, 10, 100)
    Ms = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    times_batched = []
    times_sequential = []

    for M in Ms:

        centers, radii, freqs, yaw_bools = generate_circle_params(M)
        batched_traj = BatchedThreeDCircularTraj(centers, radii, freqs, yaw_bools)
        circle_trajectories = [ThreeDCircularTraj(center=center, radius=radius, freq=freq, yaw_bool=yaw_bool) for center, radius, freq, yaw_bool in zip(centers, radii, freqs, yaw_bools)]
        sequential_trajs = BatchedTrajectory(circle_trajectories)

        x_batched, time_batched = time_traj(M, batched_traj, t)
        times_batched.append(time_batched)

        x_individual, time_sequential = time_traj(M, sequential_trajs, t)
        times_sequential.append(time_sequential)

        assert np.allclose(x_batched, x_individual), "Batched and individual trajectories do not match"

    print("Batched times: ", times_batched)
    print("Sequential times: ", times_sequential)

    fig, ax = plt.subplots()
    ax.plot(Ms, times_batched, label='Batched')
    ax.plot(Ms, times_sequential, label='Sequential')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Number of UAVs')
    ax.set_ylabel('Time (s)')
    ax.set_title('Batched vs Individual Trajectory Execution Time')
    ax.legend()

    fig.savefig(os.path.join(os.path.dirname(__file__), 'batched_traj_benchmark.png'))

if __name__=="__main__":
    benchmark_batched_circular_traj()
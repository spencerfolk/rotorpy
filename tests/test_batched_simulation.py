import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from rotorpy.trajectories.minsnap import MinSnap, BatchedMinSnap
from rotorpy.vehicles.batched_multirotor import BatchedMultirotor, merge_rotorpy_states, merge_flat_outputs
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3ControlBatch, SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.utils.trajgen_utils import sample_waypoints
from rotorpy.world import World


def main():
    device = torch.device("cpu")
    #### Initial Drone States ####
    num_drones = 1000
    init_rotor_speed = 1788.53
    x0 = {'x': torch.zeros(num_drones,3),
          'v': torch.zeros(num_drones, 3),
          'q': torch.tensor([0, 0, 0, 1]).repeat(num_drones, 1), # [i,j,k,w]
          'w': torch.zeros(num_drones, 3),
          'wind': torch.zeros(num_drones, 3),  # Since wind is handled elsewhere, this value is overwritten
          'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed]).repeat(num_drones, 1)}

    x0_single = {'x': np.array([0, 0, 0]),
     'v': np.zeros(3, ),
     'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
     'w': np.zeros(3, ),
     'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
     'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

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
    vehicle = BatchedMultirotor(quad_params, num_drones, x0)
    vehicle.motor_noise = 0
    print(f"Time to Generate reference trajectories: {time.time() - ref_traj_gen_start_time}")

    controller_single = SE3Control(quad_params)
    vehicle_single = Multirotor(quad_params, initial_state=x0_single)
    vehicle_single.motor_noise = 0

    t = 0
    dt = 0.01
    t_f = 3
    flats = [batched_trajs.update(t)]
    states = [x0]
    controls = [controller.update(states[-1], flats[-1])]

    batch_start_time = time.time()
    while t < t_f:
        t += dt
        states.append(vehicle.step(states[-1], controls[-1], dt))
        flats.append(batched_trajs.update(t))
        controls.append(controller.update(states[-1], flats[-1]))
    print(f"time to simulate {num_drones} batched: {time.time() - batch_start_time}")


    series_start_time = time.time()
    all_seq_states = []
    for d in range(num_drones):
        t = 0
        flats_s1 = [trajectories[d].update(t)]
        states_s1 = [x0_single]
        controls_s1 = [controller_single.update(t, states_s1[-1], flats_s1[-1])]
        while t < t_f:
            t += dt
            states_s1.append(vehicle_single.step(states_s1[-1], controls_s1[-1], dt))
            flats_s1.append(trajectories[d].update(t))
            controls_s1.append(controller_single.update(t, states_s1[-1], flats_s1[-1]))
        all_seq_states.append(states_s1)
    print(f"time to simulate {num_drones} sequentially: {time.time() - series_start_time}")

    num_to_plot = 3
    fig, ax = plt.subplots(num_to_plot, 3)
    which_sim = np.random.choice(num_drones, num_to_plot, replace=False)
    dims = ["x", "y", "z"]
    for j, sim_idx in enumerate(which_sim):
        for dimension in range(3):
            ax[j][dimension].plot([state['x'][sim_idx][dimension] for state in states], label='batched')
            ax[j][dimension].plot([state['x'][dimension] for state in all_seq_states[sim_idx]], label='sequential')
            ax[j][dimension].legend()
            ax[j][dimension].set_ylabel(dims[dimension])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
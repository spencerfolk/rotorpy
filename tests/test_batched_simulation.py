import numpy as np
import torch
import time

from rotorpy.trajectories.minsnap import MinSnap, BatchedMinSnap
from rotorpy.vehicles.batched_multirotor import BatchedMultirotor
from rotorpy.controllers.quadrotor_control import SE3ControlBatch
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.utils.trajgen_utils import sample_waypoints
from rotorpy.world import World


def main():
    device = torch.device("cpu")
    #### Initial Drone States ####
    num_drones = 2
    init_rotor_speed = 1788.53
    x0 = {'x': torch.zeros(num_drones,3),
          'v': torch.zeros(num_drones, 3),
          'q': torch.tensor([0, 0, 0, 1]).repeat(num_drones, 1), # [i,j,k,w]
          'w': torch.zeros(num_drones, 3),
          'wind': torch.zeros(num_drones, 3),  # Since wind is handled elsewhere, this value is overwritten
          'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed]).repeat(num_drones, 1)}

    #### Generate Trajectories ####
    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    num_waypoints = 3
    v_avg_des = 1.0
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
    print(f"Time to Generate reference trajectories: {time.time() - ref_traj_gen_start_time}")

    t = 0
    dt = 0.01
    flats = [batched_trajs.update(t)]
    states = [x0]
    controls = [controller.update(states[-1], flats[-1])]
    while t < trajectories[0].t_keyframes[-1]:
        t += dt
        states.append(vehicle.step(states[-1], controls[-1], dt))
        flats.append(batched_trajs.update(t))
        controls.append(controller.update(states[-1], flats[-1]))
        print(f"------t = {t}-------")
        print(states[-1]["x"])

if __name__ == "__main__":
    main()
import numpy as np
import torch

from rotorpy.trajectories.minsnap import BatchedMinSnap
from rotorpy.vehicles.multirotor import Multirotor, BatchedMultirotorParams, BatchedMultirotor
from rotorpy.controllers.quadrotor_control import BatchedSE3Control, SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params as cf_quad_params
from rotorpy.vehicles.hummingbird_params import quad_params as hb_quad_params
from rotorpy.utils.trajgen_utils import generate_random_minsnap_traj
from rotorpy.world import World

def test_batched_operators():
    np.random.seed(10)
    num_drones = 10
    device = torch.device("cpu")
    init_rotor_speed = 1788.53
    # Set to 0 if you want sim results to be more deterministic (default value is 100)
    cf_quad_params["motor_noise_std"] = 0
    hb_quad_params["motor_noise_std"] = 0

    # We'll simulate half crazyflies, half hummingbirds
    all_quad_params = [cf_quad_params]*(num_drones//2) + [hb_quad_params]*(num_drones//2)

    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    batch_state = {'x': torch.randn(num_drones,3, device=device).double(),
                  'v': torch.randn(num_drones, 3, device=device).double(),
                  'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_drones, 1).double(),
                  'w': torch.randn(num_drones, 3, device=device).double(),
                  'wind': torch.zeros(num_drones, 3, device=device).double(),
                  'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_drones, 1).double()}
    trajectories = [generate_random_minsnap_traj(world, 3, 1.0, 1.0, 2.0, np.random.randn(3)) for _ in range(num_drones)]
    batch_minsnap = BatchedMinSnap(trajectories, device)
    ts = np.array([np.random.uniform(0, 1) for i in range(num_drones)])
    batch_flat_output= batch_minsnap.update(ts)

    # Make sure that BatchedMinSnap does the same thing as MinSnap
    for j, traj in enumerate(trajectories):
        seq_flat_output = traj.update(ts[j])
        for key in batch_flat_output.keys():
            assert np.all(np.abs(batch_flat_output[key][j].cpu().numpy() - seq_flat_output[key]) < 1e-3)

    batch_params = BatchedMultirotorParams(all_quad_params, num_drones, device)
    batched_ctrlr = BatchedSE3Control(batch_params, num_drones, device)
    batch_control_inputs = batched_ctrlr.update(0, batch_state, batch_flat_output)

    # Make sure that BatchedSE3Control does the same thing as SE3Control
    # Make sure that BatchedMultirotor does the same thing as MultiRotor
    control_abstractions = ["cmd_motor_speeds", "cmd_motor_thrusts", "cmd_ctbm", "cmd_ctbr",
                            "cmd_ctatt", "cmd_vel", "cmd_acc"]
    for j in range(num_drones):
        single_ctrlr = SE3Control(all_quad_params[j])
        seq_state = {key: batch_state[key][j].cpu().numpy() for key in batch_state.keys()}
        flat_output = {key: batch_flat_output[key][j].cpu().numpy() for key in batch_flat_output.keys()}
        seq_control_input = single_ctrlr.update(0, seq_state, flat_output)
        for key in batch_control_inputs.keys():
            assert np.all(np.abs(batch_control_inputs[key][j].cpu().numpy() - seq_control_input[key]) < 1e-3)

    for abstraction in control_abstractions:
        print(f"Testing control abstraction = {abstraction}")
        batched_multirotor = BatchedMultirotor(batch_params, num_drones, batch_state, device, control_abstraction=abstraction)
        batch_next_state = batched_multirotor.step(batch_state, batch_control_inputs, 0.01)
        for j in range(num_drones):
            single_ctrlr = SE3Control(all_quad_params[j])
            seq_state = {key: batch_state[key][j].cpu().numpy() for key in batch_state.keys()}
            flat_output = {key: batch_flat_output[key][j].cpu().numpy() for key in batch_flat_output.keys()}
            seq_control_input = single_ctrlr.update(0, seq_state, flat_output)
            single_multirotor = Multirotor(all_quad_params[j], control_abstraction=abstraction)
            seq_next_state = single_multirotor.step(seq_state, seq_control_input, 0.01)
            for key in batch_next_state.keys():
                # since rotor speeds are large, we need a higher tolerance here.
                if key == "rotor_speeds":
                    assert np.all(np.abs(batch_next_state[key][j].cpu().numpy() - seq_next_state[key]) < 1)
                else:
                    assert np.all(np.abs(batch_next_state[key][j].cpu().numpy() - seq_next_state[key]) < 2e-2)


if __name__ == "__main__":
    test_batched_operators()

from rotorpy.vehicles.batched_multirotor import BatchedDynamicsParams
from rotorpy.vehicles.crazyflie_params import quad_params as cf_params
import numpy as np
from typing import Tuple, List, Dict

crazyflie_randomizations = {
    "mass": [0.02, 0.04],
    "k_eta": [2.3e-8, 7.0e-8],
    "tau_m": [0.003, 0.007]
}

def generate_random_dynamics_params(num_drones: int,
                                    device,
                                    nominal_params: Dict = cf_params,
                                    randomization_ranges: Dict = crazyflie_randomizations):
    batch_params = BatchedDynamicsParams([nominal_params for _ in range(num_drones)], num_drones, device)
    for idx in range(num_drones):
        update_dynamics_params(idx,
                               randomization_ranges,
                               batch_params)
    return batch_params

def generate_random_rotor_pos(num_rotors, pos_range: Tuple):
    pass

def update_dynamics_params(idx: int,
                           ranges: dict,
                           params_obj: BatchedDynamicsParams):
    if "mass" in ranges:
        params_obj.update_mass(idx, np.random.uniform(ranges["mass"][0], ranges["mass"][1]))
    if "k_eta" in ranges or "k_m" in ranges or "rotor_pos" in ranges:
        k_eta = np.random.uniform(ranges["k_eta"][0],
                                  ranges["k_eta"][1]) if "k_eta" in ranges else None
        k_m = np.random.uniform(ranges["k_m"][0],
                                ranges["k_m"][1]) if "k_m" in ranges else None
        rotor_pos=None  # not implemented yet
        params_obj.update_thrust_and_rotor_params(idx, k_eta, k_m, rotor_pos)
    if "Ixx" in ranges or "Iyy" in ranges or "Izz" in ranges:
        Ixx = np.random.uniform(ranges["Ixx"][0],
                                ranges["Ixx"][1]) if "Ixx" in ranges else None
        Iyy  = np.random.uniform(ranges["Iyy"][0],
                                 ranges["Iyy"][1]) if "Iyy" in ranges else None
        Izz  = np.random.uniform(ranges["Izz"][0],
                                 ranges["Izz"][1]) if "Izz" in ranges else None
        params_obj.update_inertia(idx, Ixx, Iyy, Izz)
    if "tau_m" in ranges:
        tau_m = np.random.uniform(ranges["tau_m"][0],ranges["tau_m"][1])
        params_obj.tau_m[idx] = tau_m
    if "motor_noise" in ranges:
        motor_noise = np.random.uniform(ranges["motor_noise"][0], ranges["motor_noise"][1])
        params_obj.motor_noise[idx] = motor_noise
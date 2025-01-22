'''
Test the Multirotor RotorPy vehicle class. 
'''

import numpy as np

def test_multirotor_class():
    
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.vehicles.crazyflie_params import quad_params

    print("\nTesting Multirotor class")

    # Create an instance of the Multirotor class
    multirotor_obj = Multirotor(quad_params)

    # Check if the class has some important attributes that are used elsewhere
    print("Checking for important attributes...")
    def check_attribute(obj, attr):
        assert hasattr(obj, attr), f"Multirotor class is missing '{attr}' attribute"
    params = ['mass', 'Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz', 
              'c_Dx', 'c_Dy', 'c_Dz', 'num_rotors', 'rotor_pos', 'rotor_dir', 
              'rotor_speed_min', 'rotor_speed_max', 'k_eta', 'k_m', 'k_d', 'k_z', 'k_flap', 
              'tau_m', 'motor_noise', 
              'inertia', 'weight', 'g', 'rotor_drag_matrix', 'drag_matrix']
    for param in params:
        check_attribute(multirotor_obj, param)

    assert hasattr(multirotor_obj, 'rotor_geometry'), "Multirotor class is missing 'rotor_geometry' attribute, is extract_geometry() no longer being called?"

    # Test if step() works. 
    print("Checking if step() method works as intended for all control abstractions...")
    assert hasattr(multirotor_obj, 'step'), "Multirotor class is missing 'step' method"

    current_state = {'x': np.zeros(3), 
                     'v': np.zeros(3), 
                     'q': np.array([0, 0, 0, 1]), 
                     'w': np.zeros(3), 
                     'wind': np.zeros(3), 
                     'rotor_speeds': np.zeros(4)}
    control_input = {'cmd_motor_speeds':np.zeros(4),
                     'cmd_motor_thrusts':np.zeros(4),
                     'cmd_thrust':0.0,
                     'cmd_moment':np.zeros(3),
                     'cmd_q':np.array([0, 0, 0, 1]),
                     'cmd_w':np.zeros(3),
                     'cmd_v':np.zeros(3),
                     'cmd_acc':np.array([1e-5, 1e-5, 1e-5])}  # Cmd acc cannot be exactly 0. 

    for control_abstraction in ['cmd_motor_speeds', 'cmd_motor_thrusts', 'cmd_ctbm', 'cmd_ctbr', 'cmd_ctatt', 'cmd_acc', 'cmd_vel']:
        print(f"\tChecking control abstraction: {control_abstraction}")
        multirotor_obj.control_abstraction = control_abstraction
        new_state = multirotor_obj.step(current_state, control_input, t_step=0.01)
        assert isinstance(new_state, dict), "step() method should return a dictionary"
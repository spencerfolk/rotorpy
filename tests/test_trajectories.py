'''
Test the trajectory module. 
'''

import inspect
import numpy as np
import glob
import os
import importlib

# The expected signature for the `update` method
EXPECTED_UPDATE_SIGNATURE = {
    'args': ['self', 't'],  # Expected arguments, 'self' is implicit
    'return_type': 'dict'  # Expected return type (dict)
}

# Define the expected structure of the returned dictionary
EXPECTED_RETURN_DICT_KEYS = {
    'x': np.ndarray,        # Position vector (m)
    'x_dot': np.ndarray,    # Velocity vector (m/s)
    'x_ddot': np.ndarray,   # Acceleration vector (m/s^2)
    'x_dddot': np.ndarray,  # Jerk vector (m/s^3)
    'x_ddddot': np.ndarray, # Snap vector (m/s^4)
    'yaw': float,           # Yaw angle (rad)
    'yaw_dot': float,       # Yaw rate (rad/s)
    'yaw_ddot': float,      # Yaw acceleration (rad/s^2)
}

def check_update_signature(class_obj):
    # Ensure the class has an 'update' method
    assert hasattr(class_obj, 'update'), f"{class_obj.__name__} does not have an 'update' method."

    # Get the signature of the 'update' method
    update_method = getattr(class_obj, 'update')

    # Get the signature of the method using inspect
    signature = inspect.signature(update_method)

    # Get the parameter names and return type
    params = list(signature.parameters.keys())  # Get parameter names as a list

    # Check if the parameter names match
    assert params == EXPECTED_UPDATE_SIGNATURE['args'], f"Expected arguments {EXPECTED_UPDATE_SIGNATURE['args']} for {class_obj.__name__}, but got {params}"

# Test case to check all trajectories adhere to input requirement. 
def test_trajectories_update_method():

    # Import all trajectories from the trajectories module
    traj_files = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'rotorpy', 'trajectories', '*.py'))
    traj_objects = []
    print("\nTesting input signatures for all trajectory classes...")
    for file in traj_files:
        traj_name = os.path.basename(file)[:-3]  # Remove '.py' extension
        if traj_name != '__init__': # Ignore the __init__.py file
            print(f"Importing trajectory classes from {traj_name}")
            traj_module = importlib.import_module(f"rotorpy.trajectories.{traj_name}")  # Import the module
            traj_classes = inspect.getmembers(traj_module, inspect.isclass)            # Get all classes in the module

            # Ignore any dependency classes in the trajectories module
            traj_objects.extend([traj_class for traj_class in traj_classes if traj_class[1].__module__ == f"rotorpy.trajectories.{traj_name}"])

    # Iterate over all imported classes from the trajectories module
    for traj_class in traj_objects:
        print(f"Checking {traj_class[0]} input signatures...")
        if hasattr(traj_class[1], 'update'):  # Check only classes with an 'update' method
            check_update_signature(traj_class[1])

# Test MinSnap trajectory class. 
def test_minsnap_traj():
    from rotorpy.trajectories.minsnap import MinSnap

    print("\nTesting MinSnap class input and output signatures")
    # Create an instance of the MinSnap class
    min_snap = MinSnap(points=np.random.rand(2, 3), 
                       yaw_angles=np.random.rand(2), 
                       yaw_rate_max=2*np.pi, 
                       poly_degree=7, 
                       yaw_poly_degree=7, 
                       v_max=3, 
                       v_avg=1, 
                       v_start=[0, 0, 0], 
                       v_end=[0, 0, 0], 
                       verbose=True)
    
    return_value = min_snap.update(0.0)
    assert isinstance(return_value, dict), "update() method should return a dictionary"

    # Check if the return value has the expected keys and types
    for key, value in EXPECTED_RETURN_DICT_KEYS.items():
        assert key in return_value, f"Expected key '{key}' in return dictionary"
        assert isinstance(return_value[key], value), f"Expected type {value} for key '{key}', but got {type(return_value[key])}"
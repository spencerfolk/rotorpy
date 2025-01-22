''' 
Test the controller module. 
'''

import inspect
import numpy as np
import glob
import os
import importlib

# The expected signature for the `update` method
EXPECTED_UPDATE_SIGNATURE = {
    'args': ['self', 't', 'state', 'flat_output'],  # Expected arguments, 'self' is implicit
    'return_type': 'dict'  # Expected return type (dict)
}

EXPECTED_RETURN_DICT = {
    'cmd_motor_speeds': np.ndarray,  # numpy array of motor speeds (e.g., [motor1_speed, motor2_speed, ...])
    'cmd_motor_thrusts': np.ndarray, # numpy array of motor thrusts (e.g., [motor1_thrust, motor2_thrust, ...])
    'cmd_thrust': float,             # scalar thrust (N)
    'cmd_moment': np.ndarray,        # numpy array of moments for each axis (e.g., [moment_x, moment_y, moment_z])
    'cmd_q': np.ndarray,             # numpy array representing a quaternion [i, j, k, w]
    'cmd_w': np.ndarray,             # numpy array of angular rates [w_x, w_y, w_z]
    'cmd_v': np.ndarray,             # numpy array representing the velocity vector [v_x, v_y, v_z]
    'cmd_acc': np.ndarray            # numpy array representing the acceleration vector [a_x, a_y, a_z]
}

# Function to check the signature of the update method
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

# Test case to check all controllers adhere to input requirement. 
def test_controllers_update_method():

    # Import all controllers from the controllers module
    controller_files = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'rotorpy', 'controllers', '*.py'))
    controller_objects = []
    print("\n")
    for file in controller_files:
        controller_name = os.path.basename(file)[:-3]  # Remove '.py' extension
        if controller_name != '__init__': # Ignore the __init__.py file
            print(f"Importing controller classes from {controller_name}")
            controller_module = importlib.import_module(f"rotorpy.controllers.{controller_name}")  # Import the module
            controller_classes = inspect.getmembers(controller_module, inspect.isclass)            # Get all classes in the module

            # Ignore any dependency classes in the controllers module
            controller_objects.extend([controller_class for controller_class in controller_classes if controller_class[1].__module__ == f"rotorpy.controllers.{controller_name}"])

    # Iterate over all imported classes from the controllers module
    for controller_class in controller_objects:
        print(f"Checking {controller_class[0]} input signatures...")
        if hasattr(controller_class[1], 'update'):  # Check only classes with an 'update' method
            check_update_signature(controller_class[1])

# Specifically test SE3Control class for both input and output signatures. 
def test_se3_control():
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.vehicles.crazyflie_params import quad_params

    print("\nTesting SE3Control class")
    # Create an instance of the SE3Control class
    se3_controller = SE3Control(quad_params)

    return_value = se3_controller.update(0.0, {'x': np.zeros(3), 
                                               'v': np.zeros(3), 
                                               'q': np.array([0, 0, 0, 1]), 
                                               'w': np.zeros(3)}, 
                                              {'x': np.zeros(3), 
                                               'x_dot': np.zeros(3),
                                               'x_ddot': np.zeros(3),
                                               'x_dddot': np.zeros(3),
                                               'x_ddddot': np.zeros(3),
                                               'yaw': 0.0, 
                                               'yaw_dot': 0.0, 
                                               'yaw_ddot': 0.0})

    # Check if the return value is a dictionary
    assert isinstance(return_value, dict), f"SE3Control 'update' method must return a dictionary, but got {type(return_value)}"
    print("SE3Control 'update' method returns a dictionary.")

    # Check if the return value has the expected keys and types
    for key, value in EXPECTED_RETURN_DICT.items():
        assert key in return_value, f"Expected key '{key}' in return dictionary"
        assert isinstance(return_value[key], value), f"Expected type {value} for key '{key}', but got {type(return_value[key])}"
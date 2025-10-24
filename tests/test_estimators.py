''' 
Test the estimator module.
'''

import inspect
import numpy as np
import glob
import os
import importlib

EXPECTED_STEP_SIGNATURE = {
    'args': ['self', 'ground_truth_state', 'controller_command', 'imu_measurement', 'mocap_measurement'],
    'return_type': 'dict'
}

EXPECTED_RETURN_DICT = {
    'filter_state': np.ndarray,     # The current filter estimate
    'covariance': np.ndarray        # The current covariance matrix
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
    assert params == EXPECTED_STEP_SIGNATURE['args'], f"Expected arguments {EXPECTED_STEP_SIGNATURE['args']} for {class_obj.__name__}, but got {params}"

# Test case to check all estimators adhere to input requirement.
def test_estimators_step_method():
    
    # Import all estimators from the estimators module
    estimator_files = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'rotorpy', 'estimators', '*.py'))
    estimator_objects = []
    print("\n")
    for file in estimator_files:
        estimator_name = os.path.basename(file)[:-3]  # Remove '.py' extension
        if estimator_name != '__init__': # Ignore the __init__.py file
            print(f"Importing estimator classes from {estimator_name}")
            estimator_module = importlib.import_module(f"rotorpy.estimators.{estimator_name}")  # Import the module
            estimator_classes = inspect.getmembers(estimator_module, inspect.isclass)

            # Ignore any dependency classes in the estimators module
            estimator_objects.extend([estimator_class for estimator_class in estimator_classes if estimator_class[1].__module__ == f"rotorpy.estimators.{estimator_name}"])

    # Iterate over all imported classes from the controllers module
    for estimator_class in estimator_objects:
        print(f"Checking {estimator_class[0]} input signatures...")
        if hasattr(estimator_class[1], 'update'):  # Check only classes with an 'update' method
            check_update_signature(estimator_class[1])
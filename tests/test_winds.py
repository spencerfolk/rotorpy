''' 
Check to confirm that the wind classes adhere to the expected signature.
'''

import inspect
import numpy as np
import glob
import os
import importlib

# The expected signature for the `update` method
EXPECTED_UPDATE_SIGNATURE = {
    'args': ['self', 't', 'position'],  # Expected arguments, 'self' is implicit
    'return_type': np.ndarray  # Expected return type (dict)
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

def test_wind_update_method():
    
    # Import all wind from the wind module
    wind_files = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'rotorpy', 'wind', '*.py'))
    wind_objects = []
    print("\nTesting input signatures for all trajectory classes...")
    for file in wind_files:
        wind_name = os.path.basename(file)[:-3]  # Remove '.py' extension
        if wind_name != '__init__': # Ignore the __init__.py file
            print(f"Importing trajectory classes from {wind_name}")
            wind_module = importlib.import_module(f"rotorpy.wind.{wind_name}")  # Import the module
            wind_classes = inspect.getmembers(wind_module, inspect.isclass)            # Get all classes in the module

            # Ignore any dependency classes in the wind module
            wind_objects.extend([wind_class for wind_class in wind_classes if wind_class[1].__module__ == f"rotorpy.wind.{wind_name}"])

    # Iterate over all imported classes from the wind module
    for wind_class in wind_objects:
        print(f"Checking {wind_class[0]} input signatures...")
        if hasattr(wind_class[1], 'update'):  # Check only classes with an 'update' method
            check_update_signature(wind_class[1])

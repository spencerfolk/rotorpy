from distutils.core import setup
from os.path import isdir
from itertools import product

# Gather our flightsim and any projXX packages that happen to exist.
all_packages = ['rotorpy']
packages = list(filter(isdir, all_packages))

setup(
    name='rotorpy',
    packages=packages,
    version='1.1.0',
    install_requires=[
            'cvxopt',
            'matplotlib',
            'filterpy == 1.4.5',
            'numpy',
            'scipy',
            'pandas',
            'ndsplines',
            'timeout_decorator',
            'tqdm',
            'gymnasium'])

import numpy as np
import os
import sys

# The Dryden Gust model is implemented using this package: 
# https://github.com/goromal/wind-dynamics
# If using the package directly, make sure the package is installed and the file ./wind-dynamics/python/wind_dynamics.so 
# exists. It may be named slightly differently depending on your system. 
# wind_path = os.path.join(os.path.dirname(__file__),'wind-dynamics/python')
# sys.path.insert(1, wind_path) # if your script is not in the wind-dynamics/python directory
# from wind_dynamics import DrydenWind

from rotorpy.wind.dryden_utils import *

class DrydenGust(object):
    """
    The Dryden Wind Turbulence model is governed by a pink noise process that is parameterized by 
    an average (mean) windspeed and standard deviation. This class is a wrapper on an 
    existing external package: https://github.com/goromal/wind-dynamics 
    The Dryden model is accepted by the DoD in design, characterization, and testing of aircraft
    in simulation. https://en.wikipedia.org/wiki/Dryden_Wind_Turbulence_Model
    """

    def __init__(self, dt=1/500, 
                    avg_wind=np.array([0,0,0]), sig_wind=np.array([1,1,1]),
                    altitude=2.0):
        """
        Inputs:
            dt := time discretization, s, should match the simulator
            avg_wind := mean windspeed on each axis, m/s
            sig_wind := turbulence intensity in windspeed on each axis, m/s
            altitude := expected operating altitude
        """
        self.dt = dt

        self.wind = DrydenWind(avg_wind[0], avg_wind[1], avg_wind[2], sig_wind[0], sig_wind[1], sig_wind[2], altitude)

    def update(self, t, position):
        """
        Given the present time and position of the multirotor, return the
        current wind speed on all three axes. 
        
        The wind should be expressed in the world coordinates. 
        """
        return self.wind.getWind(self.dt)

class DrydenGustLP(object):
    """
    This is another wrapper on an existing external package: https://github.com/goromal/wind-dynamics 
    The Dryden model is accepted by the DoD in design, characterization, and testing of aircraft
    in simulation. https://en.wikipedia.org/wiki/Dryden_Wind_Turbulence_Model

    The difference between this model and DrydenGust is that we add an additional low pass filter governed by 
    a cutoff frequency, 1/tau in order to generate smoother varying winds. 
    """

    def __init__(self, dt=1/500, 
                    avg_wind=np.array([0,0,0]), sig_wind=np.array([1,1,1]),
                    altitude=2.0,
                    tau=0.1):
        """
        Inputs:
            dt := time discretization, s, should match the simulator
            avg_wind := mean windspeed on each axis, m/s
            sig_wind := turbulence intensity (denoted sigma) in windspeed on each axis, m/s
            altitude := expected operating altitude
            tau      := cutoff frequency of the low pass filter (s)
        """
        self.dt = dt

        self.tau = tau

        self.wind = DrydenWind(avg_wind[0], avg_wind[1], avg_wind[2], sig_wind[0], sig_wind[1], sig_wind[2], altitude)
        self.prev_wind = self.wind.getWind(self.dt)

    def update(self, t, position):
        """
        Given the present time and position of the multirotor, return the
        current wind speed on all three axes. 
        
        The wind should be expressed in the world coordinates. 
        """
        wind = (1-self.dt/self.tau)*self.prev_wind + self.dt/self.tau*self.wind.getWind(self.dt)
        self.prev_wind = wind
        return wind
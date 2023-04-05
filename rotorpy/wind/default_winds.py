import numpy as np
import sys

"""
Below are some default wind objects that might be useful inputs to the system.  
"""

class NoWind(object):
    """
    This wind profile is the trivial case of no wind. It will output 
    zero wind speed on all axes for all time. 
    Alternatively, you can use ConstantWind with wx=wy=wz=0. 
    """

    def __init__(self):
        """
        Inputs: 
            Nothing
        """

    def update(self, t, position):
        """
        Given the present time and position of the multirotor, return the
        current wind speed on all three axes. 
        
        The wind should be expressed in the world coordinates. 
        """
        return np.array([0,0,0])

class ConstantWind(object):
    """
    This wind profile is constant both spatially and temporally. 
    Wind speed is specified on each axis. 
    """

    def __init__(self, wx, wy, wz):
        """
        """

        self.wind = np.array([wx, wy, wz])

        
    def update(self, t, position):
        """
        Given the present time and position of the multirotor, return the
        current wind speed on all three axes. 
        
        The wind should be expressed in the world coordinates. 
        """

        return self.wind

class SinusoidWind(object):
    """
    Wind will vary subject to a sine function with appropriate amplitude, frequency, and phase offset.
    """

    def __init__(self, amplitudes=np.array([1,1,1]), frequencies=np.array([1,1,1]), phase=np.array([0,0,0])):
        """
        Inputs: 
            amplitudes := array of amplitudes on each axis
            frequencies := array of frequencies for the wind pattern on each axis
            phase := relative phase offset on each axis in seconds
        """
        self.Ax, self.Ay, self.Az = amplitudes[0], amplitudes[1], amplitudes[2]
        self.fx, self.fy, self.fz = frequencies[0], frequencies[1], frequencies[2]
        self.px, self.py, self.pz = phase[0], phase[1], phase[2]

    def update(self, t, position):
        """
        Given the present time and position of the multirotor, return the
        current wind speed on all three axes. 
        
        The wind should be expressed in the world coordinates. 
        """

        wind = np.array([self.Ax*np.sin(2*np.pi*self.fx*(t+self.px)),
                         self.Ay*np.sin(2*np.pi*self.fy*(t+self.py)),
                         self.Az*np.sin(2*np.pi*self.fz*(t+self.pz))])

        return wind

class LadderWind(object):
    """
    The wind will step up and down between a minimum and maximum speed for a specified duration. 
    Visualized below...

                     | | <- duration
     max ->          ---         ---
                  ---         ---
               ---         ---
     min -> ---         ---
            --------------------------> t
    
    ** Normally the wind will start at min and increase sequentially, but if the random flag is set true, 
       the wind will step to a random sublevel after each duration is up. 

    """

    def __init__(self, min=np.array([-1,-1,-1]), max=np.array([1,1,1]), duration=np.array([1,1,1]), Nstep=np.array([5,5,5]), random_flag=False):
        """
        Inputs:
            min := array of minimum wind speeds across each axis
            max := array of maximum wind speeds across each axis
            duration := array of durations for each step
            Nstep := array for the integer number of discretized steps between min and max across each axis
        """

        # Check the inputs for consistency, quit and raise a flag if the inputs aren't physically realizable
        if np.any(Nstep <= 0):
            print("LadderWind Error: The number of steps must be greater than or equal to 1")
            sys.exit(1)

        if np.any(max - min < 0):
            print("LadderWind Error: The max value must be greater than the min value.")
            sys.exit(1)

        self.random_flag = random_flag
        self.durx, self.dury, self.durz = duration[0], duration[1], duration[2]
        self.nx, self.ny, self.nz = Nstep[0], Nstep[1], Nstep[2]

        # Compute arrays of intermediate wind speeds for each axis
        self.wx_arr = np.linspace(min[0], max[0], self.nx)
        self.wy_arr = np.linspace(min[1], max[1], self.ny)
        self.wz_arr = np.linspace(min[2], max[2], self.nz)

        # Initialize the amplitude id.. these numbers are used to index the arrays above to get the appropriate wind speed on each axis
        if self.random_flag:
            self.xid = np.random.choice(self.nx)
            self.yid = np.random.choice(self.ny)
            self.zid = np.random.choice(self.nz)
        else:
            self.xid, self.yid, self.zid = 0, 0, 0

        # Initialize the timers... since we don't yet know the starting time, we'll set them in the first call
        self.timerx = None
        self.timery = None
        self.timerz = None

        # Initialize the winds
        self.wx, self.wy, self.wz = self.wx_arr[self.xid], self.wy_arr[self.yid], self.wz_arr[self.zid]

    def update(self, t, position):
        """
        Given the present time and position of the multirotor, return the
        current wind speed on all three axes. 
        
        The wind should be expressed in the world coordinates. 
        """
        if self.timerx is None:
            self.timerx, self.timery, self.timerz = t, t, t
        
        if (t - self.timerx) >= self.durx:
            if self.random_flag:
                self.xid = np.random.choice(self.nx)
            else:
                self.xid = (self.xid + 1) % self.nx
            self.wx = self.wx_arr[self.xid]
            self.timerx = t

        if (t - self.timery) >= self.dury:
            if self.random_flag:
                self.yid = np.random.choice(self.ny)
            else:
                self.yid = (self.yid + 1) % self.ny
            self.wy = self.wy_arr[self.yid]
            self.timery = t

        if (t - self.timerz) >= self.durz:
            if self.random_flag:
                self.zid = np.random.choice(self.nz)
            else:
                self.zid = (self.zid + 1) % self.nz
            self.wz = self.wz_arr[self.zid]
            self.timerz = t

        return np.array([self.wx, self.wy, self.wz])
    
if __name__=="__main__":
    wind = ConstantWind(wx=1,wy=1,wz=1)
    print(wind.update(0,np.array([0,0,0])))
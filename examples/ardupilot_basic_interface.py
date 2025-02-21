"""
Ardupilot needs to receive packets from the simulator at a constant rate:
As per https://github.com/ArduPilot/ardupilot/blob/f236a6c6fcac112a2271763a344634302d65da82/libraries/SITL/examples/JSON/readme.md:

```
The frame rate represents the time step the simulation should take, 
this can be changed with the SIM_RATE_HZ ArduPilot parameter. 
The physics backend is free to ignore this value, 
a maximum time step size would typically be set. 
The SIM_RATE_HZ should value be kept above the vehicle loop rate, 
by default this 400hz on copter and quadplanes and 50 hz on plane and rover.
```

It is suggested to keep SEND_RATE_HZ at the default value of 1000HZ.
"""
from timed_count import timed_count
from rotorpy.vehicles.ardupilot_multirotor import Ardupilot, SEND_RATE_HZ
from rotorpy.vehicles.hummingbird_params import quad_params

vehicle = Ardupilot(quad_params=quad_params,ardupilot_control=True, enable_ground=True, enable_imu_noise=True)
state = vehicle.initial_state

cmd_motor_speeds = [0]*4
dt = 1/SEND_RATE_HZ

for count in timed_count(dt):
    state = vehicle.step(state, {'cmd_motor_speeds': cmd_motor_speeds}, dt)
    
    # Break after 2 seconds
    if count.time > 2:
        break
# Controller Module

The simulator is packaged with a geometric tracking controller for a quadrotor, found in `quadrotor_control.py`. Based on [this paper](https://mathweb.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf) the controller takes flat outputs (position and yaw) and outputs a dictionary containing different control abstractions (angle, rate, motor speeds). The `Multirotor` vehicle will use the commanded motor speeds in the dynamics. 

Other controllers can be developed but must complement the vehicle and the trajectory it is trying to stabilize to. 
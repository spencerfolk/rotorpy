# Vehicle Module

Vehicle classes are where the dynamics models are stored. RotorPy expects vehicles to have a `step` method, which handles dynamics integration. You can choose whatever integration method you want, although a good starting point might be `scipy.integrate.solve_ivp` [(reference)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1). 

Other sensors may require other methods for the vehicle class. For instance, the IMU sensor requires a `statedot` method, which outputs the acceleration and angular body rates of rigid body at the center of mass (the IMU sensor transforms it later on). 

Currently `Multirotor` is the only vehicle implemented with complete support (with param files, plotting and animation utilities, etc.). `Multirotor` is flexible enough to represent an arbitrary amount of rotors arbitrarily oriented relative to the body frame. It also suppports aerodynamic effects (if you can measure the relevant coefficients...). 

The following aerodynamic effects of interest are within the scope of this model: 
1. **Parasitic Drag** - Drag associated with non-lifting surfaces like the frame. This drag is quadratic in airspeed. 
2. **Rotor Drag** - This is an apparent drag force that is a result of the increased drag produced by the advancing blade of a rotor. Rotor drag is linear in airspeed. 
3. **Blade Flapping** - An effect of dissymmetry of lift, blade flapping is the motion of the blade up or down that results in a pitching moment. The pitching moment is linear in the airspeed. 
4. **Induced Drag** - Another effect of dissymmetry of lift, more apparent in semi-rigid or rigid blades, where an increase of lift on the advancing blade causes an increased induced downwash, which in turn tilts the lift vector aft resulting in more drag. Induced drag is linear in the airspeed. 
5. **Translational Lift** - In forward motion, the induced velocity at the rotor plane decreases, causing an increase in lift generation. 
6. **Translational Drag** - A consequence of translational lift, and similar to **Induced Drag**, the increased lift produced in forward flight will produce an increase in induced drag on the rotor. 

Ultimately the effects boil down to forces acting anti-parallel to the relative airspeed and a combination of pitching moments acting parallel and perpendicular to the relative airspeed for each rotor. The rotor aerodynamic effects (rotor drag, blade flapping, induced drag, and translational drag) can be lumped into a single drag force acting at each rotor hub, whereas parasitic drag can be lumped into a single force and moment vector acting at the center of mass. 

What's currently ignored: any lift produced by the frame or any torques produced by an imbalance of drag forces on the frame. We also currently neglect variations in the wind along the length of the UAV, implicitly assuming that the characteristic length scales of the wind fields are larger than UAV's maximum dimensions. Remember, the drag models used here are representative of bluff bodies, not wings. 

`Multirotor` also includes first-order motor dynamics to simulate delay/lag. 
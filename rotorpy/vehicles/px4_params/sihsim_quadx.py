import numpy as np

# 10040_sihsim_quadx preset (aligned with PX4 SIH parameters)
sihsim_quadx = {
    'mass':             1.0,     # kg      (PX4 param SIH_MASS)
    'Ixx':              0.025,   # kg·m²   (PX4 param SIH_IXX)
    'Iyy':              0.025,   # kg·m²   (PX4 param SIH_IYY)
    'Izz':              0.030,   # kg·m²   (PX4 param SIH_IZZ)
    'Ixy':              0.0,
    'Ixz':              0.0,
    'Iyz':              0.0,

    'num_rotors':       4,
    'rotor_pos': {
        'r1':           np.array([ 1.0,  1.0, 0.0]),
        'r2':           np.array([-1.0, -1.0, 0.0]),
        'r3':           np.array([ 1.0, -1.0, 0.0]),
        'r4':           np.array([-1.0,  1.0, 0.0]),
    },

    # Sign for each motor’s yaw moment
    'rotor_directions': np.array([ 1, 1, -1, -1 ]),
    'rI':               np.array([0.0, 0.0, 0.0]),

    'c_Dx':             1.0,
    'c_Dy':             1.0,
    'c_Dz':             1.0,

    'k_eta':            1.0,    # max thrust per rotor (N) → SIH_T_MAX
    'k_m':              0.1,    # max yaw moment per rotor (Nm) → SIH_Q_MAX
    'k_d':              0.0,    # rotor drag
    'k_z':              0.0,    # induced inflow
    'k_h':              0.0,    # translational lift
    'k_flap':           0.0,    # blade flapping moment

    'tau_m':            0.05,   # Motor response time constant (s) ← SIH_T_TAU
    'rotor_speed_min':  0.0,    # zero throttle
    'rotor_speed_max':  1.0,    # full throttle
    'motor_noise_std':  0.0,
}


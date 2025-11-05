# test_px4_sitl.py

from rotorpy.environments               import Environment
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.vehicles.px4_multirotor    import PX4Multirotor
from rotorpy.vehicles.px4_sihsim_quadx_params import quad_params as sihsim_quadx
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj    import HoverTraj

import numpy as np
# 1. Make sure you run px4 sitl `PX4_SIMULATOR=none PX4_SIM_MODEL=quadx make px4_sitl sihsim_quadx` in a separate terminal
# 2. Run this example
#   - python examples/basic_usage_px4.py

circular_trajectory = ThreeDCircularTraj(radius=np.array([1,1,0]))
hover_trajectory = HoverTraj(x0=np.array([0, 0, 5]))

def main():
    vehicle    = PX4Multirotor(sihsim_quadx, enable_ground=True)
    controller = SE3Control(sihsim_quadx)

    env = Environment(
        vehicle    = vehicle,
        controller = controller,
        trajectory = circular_trajectory,
        sim_rate   = 100,
    )
    results = env.run(
        t_final      = 60,
        use_mocap=False,
        plot_mocap=False,
        plot_estimator=False,
        plot_imu=False,
        plot         = True,
        animate_bool = False,
        verbose      = True,
    )

    print("Done—PX4 SITL ran for", len(results["time"]), "steps")

if __name__ == '__main__':
    main()

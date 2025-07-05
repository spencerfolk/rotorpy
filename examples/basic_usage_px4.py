# test_px4_sitl.py

from rotorpy.environments               import Environment
from rotorpy.vehicles.px4_multirotor    import PX4Multirotor, sihsim_quadx
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj    import HoverTraj

# 1. Make sure you run px4 sitl `make px4_sitl sihsim_quadx`
# 2. Run this example
#   - python examples/basic_usage_px4.py

def main():
    vehicle    = PX4Multirotor(sihsim_quadx)  
    controller = SE3Control(sihsim_quadx)
    trajectory = HoverTraj()
    env = Environment(
        vehicle    = vehicle,
        controller = controller,
        trajectory = trajectory,
        sim_rate   = 100,
    )
    results = env.run(
        t_final      = 30,
        plot         = False,
        animate_bool = False,
        verbose      = True,
    )

    print("Done—PX4 SITL ran for", len(results["time"]), "steps")

if __name__ == '__main__':
    main()

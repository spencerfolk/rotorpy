import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from rotorpy.world import World
from rotorpy.vehicles.batched_multirotor import BatchedMultirotor, BatchedDynamicsParams
from rotorpy.vehicles.crazyflie_params import quad_params as crazyflie_params
from rotorpy.learning.quadrotor_reward_functions import vec_hover_reward, vec_hover_reward_positive
from rotorpy.utils.shapes import Quadrotor
from rotorpy.learning.learning_utils import crazyflie_randomizations, update_dynamics_params

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

import math
from copy import deepcopy
from typing import Dict, Union, Any

DEFAULT_RESET_OPTIONS = {'initial_state': 'random', 'pos_bound': 2, 'vel_bound': 0,
                         "randomize_params_on_reset": False, "randomization_ranges": crazyflie_randomizations}
def _minmax_scale(x, min_values, max_values):
    '''
    Scales an array of values from
    x: np.ndarray of shape (num_envs,...) or int
    min_values: np.ndarray of shape (num_envs,...) or int
    max_values: np.ndarray of shape (num_envs,...) or int
    '''
    x_scaled = np.clip((x+1)/2 * (max_values - min_values) + min_values, min_values, max_values)
    return x_scaled


# TODO(hersh500): currently using SB3 API, but possibly better to use torchrl, RLLib, rsl-rl, or gymnasium API for GPU.
class QuadrotorVecEnv(VecEnv):
    """

    A quadrotor environment for reinforcement learning using Gymnasium.

    Inputs:
        initial_state: the initial state of the quadrotor. The default is hover.
        control_mode: the appropriate control abstraction that is used by the controller, options are...
                                    'cmd_motor_speeds': the controller directly commands motor speeds.
                                    'cmd_motor_thrusts': the controller commands forces for each rotor.
                                    'cmd_ctbr': the controller commands a collective thrsut and body rates.
                                    'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
                                    'cmd_vel': the controller commands a velocity vector in the body frame.
        reward_fn: the reward function, default to hover, but the user can pass in any function that is used as a reward.
        quad_params: the parameters for the quadrotor.
        max_time: the maximum time of the session. After this time, the session will exit.
        world: the world for the quadrotor to operate within.
        sim_rate: the simulation rate (in Hz), i.e. the timestep.
        aero: boolean, determines whether or not aerodynamic wrenches are computed.
        render_mode: render the quadrotor.
        render_fps: rendering frames per second, lower this for faster visualization.
        ax: for plotting purposes, you can supply an axis object that the quadrotor will visualize on.
        color: choose the color of the quadrotor. If none, it will randomly select a color.
    """
    metadata = {"render_modes": ["None", "3D", "console"],
                "render_fps": 30,
                "control_modes": ['cmd_motor_speeds', 'cmd_motor_thrusts', 'cmd_ctbr', 'cmd_ctbm', 'cmd_vel']}

    def __init__(self,
                 num_envs: int,
                 initial_state: Dict,
                 control_mode: str = 'cmd_vel',
                 reward_fn = vec_hover_reward,
                 quad_params: Union[Dict, BatchedDynamicsParams] = crazyflie_params,
                 device = torch.device('cpu'),
                 max_time = 10,                # Maximum time to run the simulation for in a single session.
                 wind_profile = None,         # wind profile object, if none is supplied it will choose no wind.
                 world        = None,         # The world object
                 sim_rate = 100,              # The update frequency of the simulator in Hz
                 aero = True,                 # Whether or not aerodynamic wrenches are computed.
                 render_mode = "None",        # The rendering mode
                 render_fps = 30,             # The rendering frames per second. Lower this for faster visualization.
                 fig = None,                  # Figure for rendering. Optional.
                 ax = None,                   # Axis for rendering. Optional.
                 color = None,                # The color of the quadrotor.
                 reset_options: Dict[str, Any] = DEFAULT_RESET_OPTIONS
                 ):
        self.num_envs = num_envs
        self.device = device
        self.initial_states = deepcopy(initial_state)
        self.vehicle_states = deepcopy(initial_state)
        if type(quad_params) == dict:
            self.quad_params = BatchedDynamicsParams([quad_params for _ in range(num_envs)], num_envs, device=self.device)
        else:
            self.quad_params = quad_params
        for key in self.initial_states.keys():
            self.initial_states[key] = self.initial_states[key].to(device)

        assert control_mode in self.metadata["control_modes"]  # Don't accept improper control modes
        self.control_mode = control_mode


        self.sim_rate = sim_rate
        self.t_step = 1/self.sim_rate
        self.reward_fn = reward_fn
        self.metadata["render_fps"] = render_fps

        self.quadrotor = BatchedMultirotor(batched_params=self.quad_params,
                                           num_drones = self.num_envs,
                                           initial_states=initial_state,
                                           device=self.device,
                                           control_abstraction=control_mode,
                                           aero=aero,
                                           integrator="rk4")
        self.t = np.zeros(self.num_envs)
        self.max_time = max_time

        # self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # I think stable baselines expects the action/observation space to be the same as a single env (ie. not batched)
        if self.control_mode == 'cmd_vel':
            # self.single_action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype=np.float32)
            self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype=np.float32)
        else:
            # self.single_action_space = spaces.Box(low = -1, high = 1, shape = (4,), dtype=np.float32)
            self.action_space = spaces.Box(low = -1, high = 1, shape = (4,), dtype=np.float32)

        self.rotor_speed_max = self.quad_params.rotor_speed_max.cpu().numpy()
        self.rotor_speed_min = self.quad_params.rotor_speed_min.cpu().numpy()

        # Compute the min/max thrust by assuming the rotor is spinning at min/max speed. (also generalizes to bidirectional rotors)
        self.max_thrust = self.quad_params.k_eta.cpu().numpy() * self.rotor_speed_max**2
        self.min_thrust = self.quad_params.k_eta.cpu().numpy() * self.rotor_speed_min**2

        # Find the maximum moment on each axis, N-m
        self.max_roll_moment = np.array([self.max_thrust[i] * np.abs(self.quad_params.rotor_pos[i]['r1'][1]) for i in range(self.num_envs)])
        self.max_pitch_moment = np.array([self.max_thrust[i] * np.abs(self.quad_params.rotor_pos[i]['r1'][0]) for i in range(self.num_envs)])
        self.max_yaw_moment = self.quad_params.k_m.cpu().numpy() * self.rotor_speed_max**2

        # TODO: make it so that these can be passed in?
        self.max_roll_br = 7.0
        self.max_pitch_br = 7.0
        self.max_yaw_br = 3.0

        self.max_vel = 4/math.sqrt(4)   # Selected so that at most the max speed is 4 m/s
        self.rotor_speed_order_mag = np.floor(np.log10(self.rotor_speed_max))
        self.render_mode = [render_mode] * self.num_envs
        self.reward = np.zeros(self.num_envs)

        if world is None:
            # If no world is specified, assume that it means that the intended world is free space.
            wbound = 4
            self.world = World.empty((-wbound, wbound, -wbound,
                                      wbound, -wbound, wbound))
        else:
            self.world = world

        if wind_profile is None:
            # If wind is not specified, default to no wind.
            from rotorpy.wind.default_winds import NoWind
            self.wind_profile = NoWind(num_drones=num_envs)
        else:
            self.wind_profile = wind_profile

        if render_mode == '3D':
            warnings.warn("3D Rendering in Vectorized Environment does not fully work.")
            if fig is None and ax is None:
                self.fig = plt.figure('Visualization')
                self.ax = self.fig.add_subplot(projection='3d')
            else:
                self.fig = fig
                self.ax = ax
            if color is None:
                colors = list(mcolors.CSS4_COLORS)
            else:
                colors = [color]
            self.quad_objs = [Quadrotor(self.ax, wind=True, color=np.random.choice(colors), wind_scale_factor=5) for _ in range(min(20, self.num_envs))]
            self.world_artists = None
            self.title_artist = self.ax.set_title('t = {}'.format(self.t))

        self.default_rotor_speed = 1788.54
        self.reset_options = reset_options

    def close(self):
        pass

    def reset_idx(self, env_idx, options):
        if options['initial_state'] == 'random':
            pos = torch.DoubleTensor(3, device=self.device).uniform_(-options['pos_bound'], options['pos_bound'])
            vel = torch.DoubleTensor(3, device=self.device).uniform_(-options['vel_bound'], options['vel_bound'])
            self.vehicle_states['x'][env_idx] = pos.double()
            self.vehicle_states['v'][env_idx] = vel.double()
            self.vehicle_states['q'][env_idx] = torch.tensor([0, 0, 0, 1], device=self.device).double()
            self.vehicle_states['w'][env_idx] = torch.zeros(3, device=self.device).double()
            self.vehicle_states['wind'][env_idx] = torch.zeros(3, device=self.device).double()
            self.vehicle_states['wind'][env_idx] = torch.zeros(3, device=self.device).double()
            self.vehicle_states['rotor_speeds'][env_idx] = torch.ones(4, device=self.device).double() * self.default_rotor_speed
        elif options['initial_state'] == 'deterministic':
            self.vehicle_states = self.initial_states
        elif isinstance(options['initial_state'], dict):
            # Ensure the correct keys are in dict.
            assert all(key in options['initial_state'] for key in ('x', 'v', 'q', 'w', 'wind', 'rotor_speeds'))
            for key in options['initial_state'].keys():
                self.vehicle_states[key][env_idx] = self.initial_states[key][env_idx].double().to(self.device)
        else:
            raise ValueError(
                "You must either specify 'random', 'deterministic', or provide a dict containing your desired initial state.")

        self.t[env_idx] = 0.0
        self.reward[env_idx] = 0.0

        if options["randomize_params_on_reset"]:
            update_dynamics_params(env_idx, options["randomization_ranges"], self.quad_params)

    # options is there to comply with API, but it's easier to pass it in as a class member
    def reset(self, seed=None, options=None):
        """
        Reset the environment
        Inputs:
            seed: the seed for any random number generation, mostly for reproducibility.
            options: dictionary for misc options for resetting the scene.
                        'initial_state': determines how to set the quadrotor again. Options are...
                            'random': will randomly select the state of the quadrotor.
                            'deterministic': will set the state to the initial state selected by the user when creating
                                            the quadrotor environment (usually hover).
                            the user can also specify the state itself as a dictionary... e.g.
                                reset(options={'initial_state':
                                    {'x': np.array([0,0,0]),
                                    'v': np.zeros(3,),
                                    'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                    'w': np.zeros(3,),
                                    'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                    'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
                                    })
                        'pos_bound': the min/max position region for random placement.
                        'vel_bound': the min/max velocity region for random placement

        """
        options = self.reset_options
        # If any options are not specified, set them to default values.
        if 'pos_bound' not in options:
            options['pos_bound'] = 2
        if 'vel_bound' not in options:
            options['vel_bound'] = 0
        if 'initial_state' not in options:
            options['initial_state'] = 'random'

        # Assert that the bounds are greater than or equal to 0.
        assert options['pos_bound'] >= 0 and options['vel_bound'] >= 0, "Bounds must be greater than or equal to 0."

        # Reset the gym environment
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        for i in range(self.num_envs):
            self.reset_idx(i, options)

        # Reset the time
        self.t = np.zeros(self.num_envs)

        # Reset the rewards
        self.reward = np.zeros(self.num_envs)

        # Now get observation and info using the new state
        observation = self._get_obs()
        self.reset_infos = {}

        self.render()

        return observation

    def step(self, action):
        # First rescale the action and get the appropriate control dictionary given the control mode.
        self.control_dict = self.rescale_action(action)
        for key in self.control_dict.keys():
            self.control_dict[key] = torch.from_numpy(self.control_dict[key]).to(self.device).double()

        # Now update the wind state using the wind profile
        self.vehicle_states['wind'] = self.wind_profile.update(self.t, self.vehicle_states['x'])

        # Last perform forward integration using the commanded motor speed and the current state
        self.vehicle_states = self.quadrotor.step(self.vehicle_states, self.control_dict, self.t_step)
        pre_termination_obs = self._get_obs()

        # Update t by t_step
        self.t += self.t_step

        # Check for safety
        oob = self._is_out_of_bounds()

        # Determine whether or not the session should terminate.
        time_done = self.t >= self.max_time
        dones = np.logical_or(oob, time_done)

        # Now compute the reward based on the current state
        self.reward = self._get_reward(pre_termination_obs, action)

        # Finally get info
        infos = [{} for _ in range(self.num_envs)]
        for env_idx in range(self.num_envs):
            if time_done[env_idx]:
                infos[env_idx]["terminal_observation"] = pre_termination_obs
                infos[env_idx]["TimeLimit.truncated"] = True
                self.reset_idx(env_idx, self.reset_options)
            elif oob[env_idx]:
                infos[env_idx]["terminal_observation"] = pre_termination_obs
                infos[env_idx]["TimeLimit.truncated"] = False
                self.reset_idx(env_idx, self.reset_options)

        self.render()

        observation = self._get_obs()

        return (observation, self.reward, dones, infos)

    def rescale_action(self, action):
        """
        Rescales the action to within the control limits and then assigns the appropriate dictionary, and converts to pytorch tensors.
        """

        control_dict = {}

        if self.control_mode == 'cmd_ctbm':
            # Scale action[0] to (0,1) and then scale to the max thrust
            cmd_thrust = _minmax_scale(action[...,0].reshape(-1, 1), self.quad_params.num_rotors * self.min_thrust, self.quad_params.num_rotors * self.max_thrust)

            # Scale the moments
            cmd_roll_moment = _minmax_scale(action[...,1], -self.max_roll_moment, self.max_roll_moment)
            cmd_pitch_moment = _minmax_scale(action[...,2], -self.max_pitch_moment, self.max_pitch_moment)
            cmd_yaw_moment = _minmax_scale(action[...,3], -self.max_yaw_moment, self.max_yaw_moment)

            control_dict['cmd_thrust'] = cmd_thrust
            control_dict['cmd_moment'] = np.hstack([cmd_roll_moment, cmd_pitch_moment, cmd_yaw_moment])

        elif self.control_mode == 'cmd_ctbr':
            # Scale action to min and max thrust.
            cmd_thrust = _minmax_scale(action[...,0].reshape(-1, 1), self.quad_params.num_rotors * self.min_thrust, self.quad_params.num_rotors * self.max_thrust)

            # Scale the body rates.
            cmd_roll_br = _minmax_scale(action[...,1], -self.max_roll_br, self.max_roll_br).reshape(-1, 1)
            cmd_pitch_br = _minmax_scale(action[...,2], -self.max_pitch_br, self.max_pitch_br).reshape(-1, 1)
            cmd_yaw_br = _minmax_scale(action[...,3], -self.max_yaw_br, self.max_yaw_br).reshape(-1, 1)
            control_dict['cmd_thrust'] = cmd_thrust
            control_dict['cmd_w'] = np.hstack([cmd_roll_br, cmd_pitch_br, cmd_yaw_br])

        elif self.control_mode == 'cmd_motor_speeds':
            # Scale the action to min and max motor speeds.
            control_dict['cmd_motor_speeds'] = _minmax_scale(action, self.rotor_speed_min, self.rotor_speed_max)

        elif self.control_mode == 'cmd_motor_thrusts':
            # Scale the action to min and max rotor thrusts.
            control_dict['cmd_motor_thrusts'] = _minmax_scale(action, self.min_thrust, self.max_thrust)

        elif self.control_mode == 'cmd_vel':
            # Scale the velcoity to min and max values.
            control_dict['cmd_v'] = _minmax_scale(action, -self.max_vel, self.max_vel)

        return control_dict

    def _get_reward(self, observation, action):
        return self.reward_fn(observation, action)

    def _is_out_of_bounds(self):
        se = torch.any(torch.abs(self.vehicle_states['v']) > 100, dim=-1)
        se = torch.logical_or(se, torch.any(torch.abs(self.vehicle_states['w']) > 100, dim=-1))
        se = torch.logical_or(se,
                              torch.logical_or(
                                  self.vehicle_states['x'][:,0] < self.world.world['bounds']['extents'][0],
                                  self.vehicle_states['x'][:,0] > self.world.world['bounds']['extents'][1]
                              ))
        se = torch.logical_or(se,
                              torch.logical_or(
                                  self.vehicle_states['x'][:,1] < self.world.world['bounds']['extents'][2],
                                  self.vehicle_states['x'][:,1] > self.world.world['bounds']['extents'][3]
                              ))
        se = torch.logical_or(se,
                              torch.logical_or(
                                  self.vehicle_states['x'][:,2] < self.world.world['bounds']['extents'][4],
                                  self.vehicle_states['x'][:,2] > self.world.world['bounds']['extents'][5]
                              ))

        return se.cpu().numpy()

    def _get_obs(self):
        # Concatenate all the state variables into a single vector
        state_vec = torch.cat([self.vehicle_states['x'],
                               self.vehicle_states['v'],
                               self.vehicle_states['q'],
                               self.vehicle_states['w']], dim=-1)
        return state_vec.float().cpu().numpy()

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

    # leaving mode here to comply with sb3 api, but we actually have render_mode as a class variable
    def render(self, mode=None):
        if self.render_mode[0] == '3D':
            # only plot a maximum of 20 quads, so the plot is still interpretable.
            for env_idx in range(min(self.num_envs, 20)):
                self._plot_quad(env_idx)
        elif self.render_mode[0] == 'console':
            for env_idx in range(min(self.num_envs, 10)):
                self._print_quad(env_idx)

    def _print_quad(self, env_idx):
        # print an average of the quadrotor states?
        print("env_idx: %d Time: %3.2f \t Position: (%3.2f, %3.2f, %3.2f) \t Reward: %3.2f" % (env_idx,
            self.t[env_idx], self.vehicle_states['x'][env_idx][0], self.vehicle_states['x'][env_idx][1], self.vehicle_states['x'][env_idx][2], self.reward[env_idx]))

    def _plot_quad(self, env_idx: int):
        if abs(self.t[env_idx] / (1 / self.metadata['render_fps']) - round(self.t[env_idx] / (1 / self.metadata['render_fps']))) > 5e-2:
            self.rendering = False  # Set rendering bool to false.
            return

        self.rendering = True  # Set rendering bool to true.

        plot_position = deepcopy(self.vehicle_states['x'][env_idx].cpu().numpy())
        plot_rotation = Rotation.from_quat(self.vehicle_states['q'][env_idx].cpu().numpy()).as_matrix()
        plot_wind = deepcopy(self.vehicle_states['wind'][env_idx].cpu().numpy())

        if self.world_artists is None and not ('x' in self.ax.get_xlabel()):
            self.world_artists = self.world.draw(self.ax)
            self.ax.plot(0, 0, 0, 'go')

        self.quad_objs[env_idx].transform(position=plot_position, rotation=plot_rotation, wind=plot_wind)
        self.title_artist.set_text('t = {:.2f}'.format(self.t[env_idx]))

        plt.pause(1e-9)

        return

    # The following methods are some BS to fit the stable baselines VecEnv API.
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return getattr(self, attr_name)

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return [False for i in range(self.num_envs)]

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        pass

    def step_async(self, actions: np.ndarray) -> None:
        self.async_action = actions

    def step_wait(self):
        results = self.step(self.async_action)
        self.async_action = None
        return results


def make_default_vec_env(num_envs, quad_params, controL_mode, device, **kwargs):
    num_drones = num_envs
    init_rotor_speed = 1788.53
    x0 = {'x': torch.zeros(num_drones,3, device=device).double(),
          'v': torch.zeros(num_drones, 3, device=device).double(),
          'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_drones, 1).double(),
          'w': torch.zeros(num_drones, 3, device=device).double(),
          'wind': torch.zeros(num_drones, 3, device=device).double(),
          'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_drones, 1).double()}
    return QuadrotorVecEnv(num_envs, initial_state=x0, quad_params=quad_params, max_time=5, control_mode=controL_mode, device=device, **kwargs)
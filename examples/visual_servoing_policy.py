"""
Behavior-cloning visual servoing example: land a UAV with a downward-looking
camera on a green patch hidden in a field of colored blocks, using only a
history of RGB camera images.

The policy directly outputs a velocity command (``cmd_v``) in the world frame,
which swarm vehicles apply through the multivehicle dynamics' built-in
``cmd_vel`` control abstraction. There is no low-level SE3 loop and no
min-snap trajectory generator in the loop: the goal frame is just what the
camera sees.

Data is collected by simulating a *batch* of drones at once (``simulate_batch``).
The ground-truth labels are the ``cmd_v`` commands issued by a privileged
pursuit controller, rendered offline from logged states with the batched
camera.

The demo has three stages:

  python examples/visual_servoing_policy.py generate [--num-envs 12 ...]
      Simulates a batch of drones (``--num-envs * --spawns-per-env`` episodes,
      one per drone) where a privileged pursuit controller flies each drone
      from a random spawn offset down to the green target of its environment.
      ``--num-envs`` random environments are built, and each environment spawns
      the drone ``--spawns-per-env`` times at different offsets. Logs the
      batched states, then renders the camera frames offline and writes one
      labeled npz per (environment, spawn) - the metadata records
      env_index/spawn_index so spawns of the same environment stay grouped:
      X = HISTORY_LEN-frame RGB history, Y = world-frame velocity command.

  python examples/visual_servoing_policy.py train
      Trains a CNN + MLP regressor on the saved samples. The dataset is split
      on *environment* boundaries (never between spawns of the same env), so
      validation measures generalization to whole layouts. Writes a policy
      checkpoint to <out>/policy.pt.

python examples/visual_servoing_policy.py eval [--num-envs 4]
       Flies the learned policy closed-loop on a batch of held-out
       environments (each spawned --spawns-per-env times) and reports errors
       per environment and overall. Pass --animate to also write one GIF per
       drone (3D scene + camera view) into --animate-out (default: the
       examples/ directory, next to this script).

Lower-level parameters (episode length, image size, training schedule, wind
  field, success threshold, ...) live in the "Configuration params" block at
  the top of this file and are read as defaults by the CLI; edit them rather
  than threading extra flags.

For example (from the root of the repo):

  python examples/visual_servoing_policy.py generate --num-envs 12 --spawns-per-env 4
  python examples/visual_servoing_policy.py train
  python examples/visual_servoing_policy.py eval --num-envs 4 --spawns-per-env 4

Pass --verbose to any stage for a chatty running commentary (per-env setup,
  offline render progress, per-batch losses, and per-second closed-loop updates).
"""

""" 
Imports
"""
import argparse
import collections
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import torch
from torch import nn
from torch.utils.data import DataLoader

from rotorpy.world import World
from rotorpy.world_features import registered_mode_names
from rotorpy.sensors.camera import BatchedPinholeCamera
from rotorpy.sensors.imu import BatchedImu
from rotorpy.wind.dryden_winds import BatchedDrydenGust
from rotorpy.trajectories.hover_traj import BatchedHoverTraj
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import BatchedMultirotorParams, BatchedMultirotor
from rotorpy.simulate import simulate_batch

""" 
Configuration params
"""

# World / task geometry 
GRID_N = 10                # number of cuboids along each axis
CUBOID = 1.0               # size of the cuboids
BLOCK_TOP = 1.0            # z location of the top face of the cuboids
TARGET_COLOR = [0.0, 1.0, 0.0]
START_Z = 10.0             # absolute spawn altitude
SPAWN_MIN_DIST = 0.0       # min horizontal spawn distance from the target center (m)
SPAWN_MAX_DIST = 6.0       # max horizontal spawn distance from the target center (m);
                           # spawns are additionally kept inside the map bounds
HOVER_Z = 1.0              # hover clearance (m) above the target block top
HOVER_ALTITUDE = BLOCK_TOP + HOVER_Z  # absolute hover altitude (m), used internally
SUCCESS_THRESHOLD = 0.6    # eval: horizontal error (m) that counts as a success
                           # during the final hover

# Block features (what the downward camera sees) 
# The rotorpy feature system (rotorpy/world_features.py) registers generators
# by name; the demo applies the chosen generator to every block of every env.
# Only the placement knob that matches FEATURE_MODE matters, the rest are
# ignored. To use your own generator, register it there and select it here.
FEATURE_MODE = 'random'   # built-ins: 'regular', 'random' (alias 'density'),
                           # 'edge_uniform', 'edge_random', or a custom
                           # registered generator; run the demo to see the
                           # exact list (raised error lists all of them)
FEATURE_SPACING = 0.2      # m between features      ('regular', 'edge_uniform')
FEATURE_DENSITY = 100.0     # features per m^2        ('random' / 'density')
EDGE_SPACING = 0.2         # m between features      ('edge_uniform')
EDGE_DENSITY = 50.0        # features per m of edge  ('edge_random')
FEATURE_NOISE = 0.0        # std of Gaussian RGB jitter applied per feature
                           # (features otherwise take their block's color)

# Camera and control loop 
H_LEN = 10                 # frames of history the policy sees
HISTORY_LEN = H_LEN
FRAME_RATE = 10            # camera frames/second the policy labels are drawn from
SIM_RATE = 100             # simulator update rate, Hz
IMG = 128                  # image resolution (square)
SPLAT = 2                  # splat radius (pixels) of a rendered feature point
VEL_SCALE = 3.0            # velocity commands scaled by this at train time and
                           # clamped to +-VEL_SCALE during collection/eval
SAFETY_MARGIN = 0.25       # collision/sphere radius for the batched simulator
KP = [0.8, 0.8, 2.0]       # pursuit gains of the privileged expert
                           # (world frame, m/s per m)

# Wind field 
# Each environment's drone gets a random horizontal *mean* wind whose magnitude
# is drawn from the [WIND_MIN_SPEED, WIND_MAX_SPEED] range, plus a shared
# Dryden turbulent gust component (WIND_GUST_SIG per axis) on top. Zero the
# gust sigmas to disable the turbulence; set both speed bounds to zero for a
# calm environment, or grow the range for harsher conditions.
WIND_MIN_SPEED = 0.5               # smallest mean wind magnitude (m/s)
WIND_MAX_SPEED = 3.0               # largest mean wind magnitude (m/s)
WIND_GUST_SIG = (20.0, 20.0, 40.0)  # Dryden turbulence intensity (m/s) per axis

# Data collection / training / eval defaults 
NUM_ENVS = 12              # environments simulated in one generate batch
SPAWNS_PER_ENV = 1         # spawn points drawn per environment
ENV_OFFSET = 0             # first environment index used by generate
SEED = 1                   # RNG seed for world/spawn generation and eval
T_FINAL = 8.0              # episode length during data collection (s)
EPOCHS = 20                # training epochs
BATCH_SIZE = 64            # training batch size
LR = 1e-3                  # Adam learning rate
VAL_FRACTION = 0.15        # fraction of *environments* held out for validation
SEED_TRAIN = 0             # RNG seed for training
EVAL_NUM_ENVS = 4          # environments evaluated in one batch
EVAL_ENV_OFFSET = 1000     # eval starts at this env index (held-out for the policy)
EVAL_T_FINAL = 12.0        # closed-loop eval horizon (s)

# Output locations
DATA_OUT = "examples/visual_servoing_data"
MODEL_OUT = "examples/visual_servoing_models"
PRINT_FPS = False          # print the batched simulation frame rate
ANIMATE_OUT = "examples/visual_servoing_animations"  # eval GIFs go here

# Downward-looking camera rig (see IMG/SPLAT above).
DOWN_ORIENTATION = Rotation.from_euler('x', 180, degrees=True).as_quat()
INTRINSICS = dict(width=IMG, height=IMG, fx=55.0, fy=55.0,
                  cx=(IMG - 1) / 2.0, cy=(IMG - 1) / 2.0,
                  dist_coeffs=np.zeros(5))

# Small list of other colors to draw from to distract the policy.
DISTRACTOR_PALETTE = [
    [0.9, 0.1, 0.1], [0.1, 0.1, 0.9], [1.0, 0.6, 0.0], [0.6, 0.1, 0.7],
    [0.0, 0.7, 0.8], [0.9, 0.8, 0.1], [0.9, 0.2, 0.6], [0.5, 0.5, 0.5],
]

""" 
Utility/helper scripts
"""

def _hover_rotor_speed(quad_params):
    return float(np.sqrt(quad_params['mass'] * 9.81 /
                         (quad_params['num_rotors'] * quad_params['k_eta'])))

def _resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_world(rng, env_index):
    """
    Build a grid of cuboids on the floor. One random cuboid is the
    target, the others get random colors. Surface features are placed with
    the FEATURE_MODE generator from the config block.
    Returns (World, target_center_xy).
    """
    indices = [(i, j) for i in range(GRID_N) for j in range(GRID_N)]
    ti, tj = indices[int(rng.integers(0, len(indices)))]
    blocks = []
    for (i, j) in indices:
        color = TARGET_COLOR if (i, j) == (ti, tj) \
            else list(DISTRACTOR_PALETTE[int(rng.integers(0, len(DISTRACTOR_PALETTE)))])
        blocks.append({'extents': [i * CUBOID, (i + 1) * CUBOID,
                                   j * CUBOID, (j + 1) * CUBOID,
                                   0.0, BLOCK_TOP],
                       'color': color})
    world_data = {'bounds': {'extents': [-3.0, GRID_N * CUBOID + 3.0,
                                         -3.0, GRID_N * CUBOID + 3.0,
                                          0.0, START_Z + 3.0]},
                  'blocks': blocks}
    if FEATURE_MODE not in registered_mode_names():
        raise ValueError("Unknown FEATURE_MODE %r in the config block; "
                         "available generators: %s"
                         % (FEATURE_MODE, ", ".join(registered_mode_names())))
    world = World(world_data, add_features=True,
                  feature_mode=FEATURE_MODE,
                  feature_spacing=FEATURE_SPACING,
                  feature_density=FEATURE_DENSITY,
                  edge_spacing=EDGE_SPACING,
                  edge_density=EDGE_DENSITY,
                  descriptor_noise=FEATURE_NOISE,
                  seed=int(env_index) * 7919 + 1)
    return world, np.array([ti + 0.5, tj + 0.5])

def spawn_offset(rng, target_xy, xy_min, xy_max):
    """
    Random horizontal displacement of the spawn from the target center, with
    the radial distance drawn uniformly from [SPAWN_MIN_DIST, SPAWN_MAX_DIST]
    (config block). Rejection-samples so the resulting position stays inside
    the map bounds [xy_min, xy_max] per axis - the target can sit near the
    field edge, so part of the annulus is off the map.
    """
    for _ in range(64):
        rho = rng.uniform(SPAWN_MIN_DIST, SPAWN_MAX_DIST)
        theta = rng.uniform(0.0, 2 * np.pi)
        p = target_xy + rho * np.array([np.cos(theta), np.sin(theta)])
        if bool(np.all((p >= xy_min) & (p <= xy_max))):
            return p - target_xy
    # The full ring is off the map (or the config is degenerate): fall back to
    # the min-distance point aimed at the map center, clipped into the bounds.
    center = 0.5 * (xy_min + xy_max)
    direction = center - target_xy
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm
    p = np.clip(target_xy + direction * SPAWN_MIN_DIST, xy_min, xy_max)
    return p - target_xy

def build_envs(rng, num_envs, env_offset):
    """
    Build `num_envs` independent random environments. Returns (env_worlds,
    env_targets) where env_targets[e] is the (x, y) center of the green target
    of environment e.
    """
    env_worlds, env_targets = [], []
    for e in range(num_envs):
        world, target_xy = build_world(rng, env_offset + e)
        env_worlds.append(world)
        env_targets.append(target_xy)
    return env_worlds, np.array(env_targets)

def spawn_batch(rng, env_targets, num_spawns, env_worlds):
    """
    Expand E environments into E * num_spawns batch drones: each of the
    `num_spawns` distinct random spawn offsets is drawn per environment.
    Returns (worlds, starts, env_idx, spawn_idx) where the per-drone world and
    starting position are paired with 0-based batch-relative env/spawn indices
    (drone b belongs to environment env_idx[b], spawn spawn_idx[b]).
    """
    if not (0.0 <= SPAWN_MIN_DIST <= SPAWN_MAX_DIST):
        sys.exit("SPAWN_MIN_DIST (%.2f) must satisfy 0 <= min <= SPAWN_MAX_DIST "
                 "(%.2f); edit the config block" % (SPAWN_MIN_DIST, SPAWN_MAX_DIST))
    num_envs = len(env_targets)
    worlds = [w for w in env_worlds for _ in range(num_spawns)]
    env_idx = np.repeat(np.arange(num_envs), num_spawns)
    spawn_idx = np.tile(np.arange(num_spawns), num_envs)
    xy_min = np.array([0.0, 0.0])
    xy_max = np.array([GRID_N * CUBOID, GRID_N * CUBOID])
    starts = []
    for e in range(num_envs):
        for _ in range(num_spawns):
            start = np.append(env_targets[e], START_Z)
            start[:2] += spawn_offset(rng, env_targets[e], xy_min, xy_max)
            starts.append(start)
    return worlds, np.stack(starts), env_idx, spawn_idx

def make_camera(num_drones, device):
    """
    A batched downward-looking camera shared by all drones in the batch.
    """
    return BatchedPinholeCamera(num_drones,
                                intrinsics=dict(INTRINSICS),
                                extrinsics={'position': np.zeros(3),
                                            'orientation': DOWN_ORIENTATION},
                                near_plane=0.2, device=device,
                                splat_radius=SPLAT, feature_output='rgb')

def make_initial_states(starts, device):
    """
    Batched initial states for drones spawned at the given (B, 3) positions.
    """
    B = starts.shape[0]
    rotor0 = _hover_rotor_speed(quad_params)
    return {'x': torch.tensor(starts, dtype=torch.double, device=device),
            'v': torch.zeros(B, 3, dtype=torch.double, device=device),
            'q': torch.tensor([0., 0., 0., 1.], dtype=torch.double,
                              device=device).repeat(B, 1),
            'w': torch.zeros(B, 3, dtype=torch.double, device=device),
            'wind': torch.zeros(B, 3, dtype=torch.double, device=device),
            'rotor_speeds': torch.full((B, quad_params['num_rotors']), rotor0,
                                       dtype=torch.double, device=device)}

def make_wind(rng, num_drones, device):
    """
    Per-drone biased gusty wind: a random horizontal *mean* wind per drone whose
    magnitude is drawn from [WIND_MIN_SPEED, WIND_MAX_SPEED], plus the shared
    Dryden turbulence component (WIND_GUST_SIG). Aggressive enough to knock
    the drone off the pursuit line during data collection.
    """
    avgs, sigs = [], []
    for _ in range(num_drones):
        speed = rng.uniform(WIND_MIN_SPEED, WIND_MAX_SPEED)
        angle = rng.uniform(0.0, 2 * np.pi)
        avg = speed * np.array([np.cos(angle), np.sin(angle), 0.0])
        avgs.append(avg)
        sigs.append(np.array(WIND_GUST_SIG))
    return BatchedDrydenGust(dt=1.0 / SIM_RATE,
                             avg_wind=torch.tensor(np.stack(avgs),
                                                   dtype=torch.double,
                                                   device=device),
                             sig_wind=torch.tensor(np.stack(sigs),
                                                   dtype=torch.double,
                                                   device=device),
                             altitude=HOVER_ALTITUDE, device=device)

def _never_exit(t, states):
    """
    terminate callback for simulate_batch: never exit before t_final, so every
    drone runs for the full episode duration.
    """
    return np.zeros(len(states['x']), dtype=bool)

""" 
Policies/controllers
"""

class CmdVelExpert(object):
    """
    Privileged expert: a simple pursuit controller that knows the target
    location. Outputs the world-frame velocity command the multivehicle
    dynamics apply directly through the 'cmd_vel' control abstraction.
    """

    def __init__(self, num_drones, targets, kp, device):
        self.num_drones = num_drones
        self.targets = torch.tensor(targets, dtype=torch.double, device=device)
        self.kp = torch.tensor(kp, dtype=torch.double, device=device)

    def update(self, t, states, flat_outputs, idxs=None):
        if idxs is None:
            idxs = list(range(self.num_drones))
        idxs = list(idxs)
        cmd = torch.clamp(self.kp * (self.targets[idxs] - states['x'][idxs]),
                          -VEL_SCALE, VEL_SCALE)
        out = torch.zeros(self.num_drones, 3, dtype=torch.double,
                          device=states['x'].device)
        out[idxs] = cmd
        return {'cmd_v': out}


class PolicyController(object):
    """
    Batched learned controller: at the camera frame rate it renders a fresh
    image of each running drone, pushes it onto that drone's history buffer,
    and asks the policy for a velocity. The predicted velocity is emitted
    directly as the world-frame 'cmd_v' command - the same quantity the expert
    was trained to imitate.
    """

    def __init__(self, policy, camera, worlds, histories, device, verbose=False):
        self.policy = policy
        self.camera = camera
        self.worlds = worlds
        self.num_drones = len(worlds)
        self.device = device
        self.histories = histories
        self.verbose = verbose
        self.period = 1.0 / FRAME_RATE
        self.v = torch.zeros(self.num_drones, 3, dtype=torch.double,
                             device=device)
        self._next_render_t = 0.0
        self._next_report_t = 0.0
        self._n_renders = 0

    def _predict(self, idx):
        windows = torch.stack([torch.stack(list(self.histories[i]))
                               for i in idx])            # (K, T, H, W, 3) uint8
        obs = windows.float().div(255.0).permute(0, 1, 4, 2, 3)  # (K, T, 3, H, W)
        with torch.no_grad():
            return torch.clamp(self.policy(obs) * VEL_SCALE,
                               -VEL_SCALE, VEL_SCALE)

    def update(self, t, states, flat_outputs, idxs=None):
        if idxs is None:
            idxs = list(range(self.num_drones))
        idxs = list(idxs)
        t_now = float(np.min(np.atleast_1d(t)))
        if t_now >= self._next_render_t - 1e-6:
            xq = {'x': torch.stack([states['x'][i] for i in idxs]).double(),
                  'q': torch.stack([states['q'][i] for i in idxs]).double()}
            img = self.camera.render([self.worlds[i] for i in idxs], xq)['image']
            img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            for j, i in enumerate(idxs):
                self.histories[i].append(img[j])
            self.v[idxs] = self._predict(idxs).double()
            if self.verbose and t_now >= self._next_report_t - 1e-6:
                print("  [eval] t=%5.1f s  frames rendered: %3d  drones running: "
                      "%d/%d" % (t_now, self._n_renders + 1, len(idxs),
                                 self.num_drones))
                self._next_report_t = t_now + 1.0
            self._next_render_t = t_now + self.period
            self._n_renders += 1
        out = torch.zeros(self.num_drones, 3, dtype=torch.double,
                          device=states['x'].device)
        out[idxs] = self.v[idxs]
        return {'cmd_v': out}

""" 
Dataset generation
"""

def render_episode_logs(camera, worlds, state, control, exit_timesteps,
                        device, t_final, verbose=False):
    """
    Re-render the batched camera frames offline at FRAME_RATE from the logged
    states. Returns per-drone lists of (uint8 frame, velocity command at that
    time), so no camera work slows down the dynamics simulation.
    """
    cam_dt = 1.0 / FRAME_RATE
    sim_dt = 1.0 / SIM_RATE
    B = state['x'].shape[1]
    n_frames = int(np.ceil(t_final / cam_dt))
    if verbose:
        print("Rendering %d camera frames offline at %d Hz from the logged "
              "states ..." % (n_frames, FRAME_RATE))
    frames = [[] for _ in range(B)]
    cmds = [[] for _ in range(B)]
    for k in range(n_frames):
        si = int(round(k * cam_dt / sim_dt))
        running = [b for b in range(B) if si < int(exit_timesteps[b])]
        if verbose and (k % 10 == 0 or len(running) == 0):
            print("  [render] frame %3d/%3d  drones still running: %d/%d"
                  % (k + 1, n_frames, len(running), B))
        if not running:
            break
        xq = {'x': torch.tensor(np.stack([state['x'][si, b] for b in running]),
                                dtype=torch.double, device=device),
              'q': torch.tensor(np.stack([state['q'][si, b] for b in running]),
                                dtype=torch.double, device=device)}
        img = camera.render([worlds[b] for b in running], xq)['image']
        img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        for j, b in enumerate(running):
            frames[b].append(img[j])
            cmds[b].append(control['cmd_v'][si, b])
    return frames, cmds


def generate_main(args):
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    # One environment per green target layout; each environment spawns the
    # drone `spawns_per_env` times at different offsets.
    env_worlds, env_targets = build_envs(rng, args.num_envs, ENV_OFFSET)
    worlds, starts, env_idx, spawn_idx = spawn_batch(
        rng, env_targets, args.spawns_per_env, env_worlds)
    B = len(worlds)
    targets_xy = env_targets[env_idx]
    targets = np.hstack([targets_xy, np.full((B, 1), HOVER_ALTITUDE)])

    if args.verbose:
        for e in range(args.num_envs):
            print("  [gen] env %03d: target (%.2f, %.2f)  %d spawn(s)"
                  % (ENV_OFFSET + e, env_targets[e, 0], env_targets[e, 1],
                     args.spawns_per_env))
            for s in range(args.spawns_per_env):
                b = e * args.spawns_per_env + s
                print("    spawn %d: start (%.2f, %.2f, %.1f)"
                      % (s, starts[b, 0], starts[b, 1], starts[b, 2]))

    x0 = make_initial_states(starts, device)
    batch_params = BatchedMultirotorParams([quad_params] * B, B, device)
    vehicle = BatchedMultirotor(batch_params, B, x0, device=device,
                                control_abstraction='cmd_vel', integrator='rk4')
    controller = CmdVelExpert(B, targets, KP, device)
    traj = BatchedHoverTraj(B, x0=targets, device=device)
    wind = make_wind(rng, B, device)
    imu = BatchedImu(B, device=device)

    # simulate_batch checks collisions against the per-drone worlds when it is
    # passed a list of Worlds (one per drone) instead of a single shared world.

    print("Simulating %d drones (%d environments x %d spawns each) for %.1f s ..."
          % (B, args.num_envs, args.spawns_per_env, T_FINAL))
    _, state, control, _, _, _, _, exit_timesteps, _ = simulate_batch(
        worlds, x0, vehicle, controller, traj, wind, imu,
        np.full(B, T_FINAL), 1.0 / SIM_RATE,
        SAFETY_MARGIN, terminate=_never_exit, print_fps=PRINT_FPS)

    # Re-render the camera frames offline from the logged states.
    camera = make_camera(B, device)
    frames, cmds = render_episode_logs(camera, worlds, state, control,
                                       exit_timesteps, device, T_FINAL,
                                       verbose=args.verbose)

    os.makedirs(args.out, exist_ok=True)
    n_episodes, n_samples = 0, 0
    for b in range(B):
        n_total = len(frames[b])
        env_abs = ENV_OFFSET + int(env_idx[b])
        sp = int(spawn_idx[b])
        if n_total - HISTORY_LEN + 1 < 10:
            print("  [gen] env %03d spawn %d: only %d frames; skipping"
                  % (env_abs, sp, n_total))
            continue
        # Store the raw frames and let the training dataset window them on the
        # fly (instead of overlapping pre-windowed arrays): ~10x less data on
        # disk and in memory, and the files can be memory-mapped during
        # training instead of being loaded into RAM.
        images = np.stack(frames[b])
        labels = np.stack(cmds[b][HISTORY_LEN - 1:])
        meta = dict(env_index=env_abs, spawn_index=sp,
                    num_spawns=args.spawns_per_env,
                    target_xy=env_targets[env_idx[b]], start_xyz=starts[b],
                    final_xyz=targets[b], n_frames=int(images.shape[0]),
                    window_len=HISTORY_LEN)
        ep_path = os.path.join(args.out, "episode_%03d_%d.npz" % (env_abs, sp))
        np.savez(ep_path, images=images, cmd_v=labels, **meta)
        n_episodes += 1
        n_samples += n_total - HISTORY_LEN + 1
        print("  [gen] saved episode_%03d_%d.npz: %d samples -> %s"
              % (env_abs, sp, n_total - HISTORY_LEN + 1,
                 os.path.abspath(ep_path)))

        if args.preview and b == 0:
            px = os.path.join(args.out, "_preview")
            os.makedirs(px, exist_ok=True)
            for k in np.linspace(0, n_total - 1, 10).astype(int):
                plt.imsave(os.path.join(px, "episode_%03d_frame_%04d.png"
                                        % (env_abs, k)), frames[b][k])
            if args.verbose:
                print("  [gen] wrote 10 preview frames to %s" % os.path.abspath(px))

    print("Wrote %d episodes (%d samples total, %d environments x %d spawns) to %s"
          % (n_episodes, n_samples, args.num_envs, args.spawns_per_env,
             os.path.abspath(args.out)))
    print("Each sample is a %d-frame x %dx%d RGB history plus a world-frame "
          "velocity command." % (HISTORY_LEN, IMG, IMG))

""" 
Training
"""

def build_policy():

    class FrameCNN(nn.Module):
        def __init__(self):
            super(FrameCNN, self).__init__()
            self.body = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            )
            # Keep coarse spatial bins (2x2) so the head can read the *bearing* to the
            # green target; a single global average pool discards exactly the
            # directional information a servoing policy needs.
            self.pool = nn.AdaptiveAvgPool2d(2)

        def forward(self, x):
            return self.pool(self.body(x)).flatten(1)  # (B, 64 * 2 * 2)

    class VelPolicy(nn.Module):
        def __init__(self):
            super(VelPolicy, self).__init__()
            self.cnn = FrameCNN()
            self.head = nn.Sequential(
                nn.Linear(64 * 4 * HISTORY_LEN, 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, 128), nn.ReLU(),
                nn.Linear(128, 3))

        def forward(self, obs):
            B, T, C, H, W = obs.shape
            feats = self.cnn(obs.reshape(B * T, C, H, W)).reshape(B, T * 64 * 4)
            return self.head(feats)

    return VelPolicy()


class _WindowDataset(object):
    """
    Random 10-frame windows sampled from within single episodes, with a
    light photometric jitter so the policy learns the green target by color
    rather than by memorizing exact pixel values of the training worlds.

    Episodes written by 'generate' are stored as raw frame series and are
    memory-mapped here (np.savez -> np.load(mmap_mode='r')), so the dataset
    never holds the whole set in heap RAM - only the pages the OS chooses to
    cache - and the HISTORY_LEN windows are sliced out on the fly. Legacy
    files saved as overlapping windows (a 'frames' key) are still accepted,
    but load fully into memory.
    """

    def __init__(self, episode_files):
        self._assets, self.bounds = [], [0]
        for f in episode_files:
            try:
                d = np.load(f, mmap_mode='r')
            except ValueError:
                d = np.load(f)     # legacy compressed npz: cannot mmap
            if 'images' in d:
                imgs, cmds = d['images'], d['cmd_v']   # (K,H,W,3) + (K-9,3)
                count = imgs.shape[0] - HISTORY_LEN + 1
                kind = 'series'
            else:
                imgs, cmds = d['frames'], d['cmd_v']   # (S,T,H,W,3) + (S,3)
                count = imgs.shape[0]
                kind = 'windows'
            self._assets.append((kind, imgs, cmds))
            self.bounds.append(self.bounds[-1] + count)
        self.n = self.bounds[-1]
        self._ep_of = np.repeat(np.arange(len(self._assets)),
                                np.diff(self.bounds))
        self._aug = np.random.default_rng()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        e = self._ep_of[idx]
        k = idx - self.bounds[e]
        kind, imgs, cmds = self._assets[e]
        if kind == 'series':
            frames = imgs[k:k + HISTORY_LEN]   # copied from the memory map
        else:
            frames = imgs[k]
        frames = frames.astype(np.float32) / 255.0     # (T, H, W, 3)
        gain = self._aug.uniform(0.85, 1.15)
        contrast = self._aug.uniform(0.9, 1.1)
        frames = np.clip((frames - 0.5) * contrast * gain + 0.5, 0.0, 1.0)
        obs = frames.transpose(0, 3, 1, 2)                       # (T, 3, H, W)
        label = (cmds[k] / VEL_SCALE).astype(np.float32)
        return obs, label


def train_main(args):

    if not os.path.isdir(args.dataset):
        sys.exit("dataset directory not found: %s (run the 'generate' stage first)"
                 % args.dataset)
    files = sorted(os.path.join(args.dataset, f)
                   for f in os.listdir(args.dataset)
                   if f.endswith(".npz"))
    if not files:
        sys.exit("no episode npz files in %s" % args.dataset)

    # Group the episodes by environment (data was generated with --spawns-per-env):
    # spawns of the same environment must never straddle the train/val split, or
    # the val score would be inflated by near-duplicate trajectories.
    env_of = {}
    legacy_files = 0
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            env_of[f] = int(d['env_index'])
            legacy_files += int('frames' in d and 'images' not in d)
    envs_sorted = sorted(set(env_of.values()))
    n_val_envs = max(1, int(len(envs_sorted) * VAL_FRACTION))
    val_envs = set(envs_sorted[-n_val_envs:])
    train_files = [f for f in files if env_of[f] not in val_envs]
    val_files = [f for f in files if env_of[f] in val_envs]
    if not train_files:
        sys.exit("no train environments left after holding out %d of %d: "
                 "generate more (--num-envs/--spawns-per-env) or lower "
                 "VAL_FRACTION in the config" % (len(val_envs), len(envs_sorted)))
    print("Dataset: %d train episodes from %d env(s), %d val episodes from %d "
          "env(s)" % (len(train_files), len(envs_sorted) - len(val_envs),
                      len(val_files), len(val_envs)))
    train_set = _WindowDataset(train_files)
    val_set = _WindowDataset(val_files)
    if legacy_files:
        print("Note: %d episode(s) use the legacy overlapping-window format, "
              "which loads fully into RAM. Re-run 'generate' to rewrite them "
              "in the streamed memory-mapped format." % legacy_files)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    torch.manual_seed(SEED_TRAIN)
    device = _resolve_device(args.device)
    policy = build_policy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val, best_state = float('inf'), None

    if args.verbose:
        print("  train set: %d samples over %d episodes (%d batches)"
              % (len(train_set), len(train_files), len(train_loader)))
        print("  val set:   %d samples over %d episodes (%d batches)"
              % (len(val_set), len(val_files), len(val_loader)))
        for e in envs_sorted:
            print("  env %03d [%s]: %d episode(s)"
                  % (e, "val" if e in val_envs else "train",
                     sum(1 for f in files if env_of[f] == e)))
        print("  device %s  %d trainable parameters  lr %.1e"
              % (device, sum(p.numel() for p in policy.parameters()), LR))

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        policy.train()
        train_loss, n = 0.0, 0
        for bi, (obs, label) in enumerate(train_loader):
            obs, label = obs.to(device), label.to(device)
            opt.zero_grad()
            loss = criterion(policy(obs), label)
            loss.backward()
            opt.step()
            train_loss += loss.item() * label.shape[0]
            n += label.shape[0]
            if args.verbose and (bi + 1) % max(1, len(train_loader) // 10) == 0:
                print("  [train] epoch %d/%d batch %d/%d loss %.5f"
                      % (epoch + 1, args.epochs, bi + 1, len(train_loader),
                         loss.item()))
        policy.eval()
        val_loss, nv = 0.0, 0
        with torch.no_grad():
            for vi, (obs, label) in enumerate(val_loader):
                obs, label = obs.to(device), label.to(device)
                val_loss += criterion(policy(obs), label).item() * label.shape[0]
                nv += label.shape[0]
                if args.verbose and (vi + 1) % max(1, len(val_loader) // 5) == 0:
                    print("    [val] epoch %d/%d batch %d/%d loss %.5f"
                          % (epoch + 1, args.epochs, vi + 1, len(val_loader),
                             (val_loss / nv)))
        print("epoch %2d/%d  train_mse %.5f  val_mse %.5f  (%.1f s)"
              % (epoch + 1, args.epochs, train_loss / max(n, 1),
                 val_loss / max(nv, 1), time.perf_counter() - t0))

        # Per-epoch checkpoint, written only when the validation loss improves.
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone()
                          for k, v in policy.state_dict().items()}
            ckpt = os.path.join(args.out, "policy_epoch_%03d.pt" % (epoch + 1))
            os.makedirs(args.out, exist_ok=True)
            torch.save(best_state, ckpt)
            print("  [ckpt] epoch %d: val_mse %.5f (best) -> %s"
                  % (epoch + 1, val_loss / max(nv, 1),
                     os.path.abspath(ckpt)))

    os.makedirs(args.out, exist_ok=True)
    torch.save(best_state if best_state is not None else policy.state_dict(),
               os.path.join(args.out, "policy.pt"))
    print("Saved policy to %s/policy.pt" % os.path.abspath(args.out))

""" 
Evaluation
"""

def _camera_pose_from_state(extrinsics, x, q_wb):
    """
    Camera world pose matching the one BatchedPinholeCamera.render() builds
    internally (p_WC = x + R_WB @ p_BC, q_WC = q_BC * q_WB^{-1}), so the
    frustum drawn in the 3D scene lines up with the rendered camera view.
    """
    p_BC = np.asarray(extrinsics['position'], dtype=np.float64)
    q_BC = np.asarray(extrinsics['orientation'], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    R_WB = Rotation.from_quat(np.asarray(q_wb, dtype=np.float64)).as_matrix()
    p_WC = x + R_WB @ p_BC
    q_WC = (Rotation.from_quat(q_BC) * Rotation.from_quat(q_wb).inv()).as_quat()
    return {'x': p_WC, 'q': q_WC}

def animate_eval_runs(camera, worlds, env_idx, spawn_idx, state,
                      exit_timesteps, t_final, device, out_dir, env_offset,
                      verbose=False, fps=FRAME_RATE):
    """
    Write one GIF per eval drone after a closed-loop eval, showing the 3D scene
    (blocks, features, drone, downward camera frustum) next to the camera's own
    view, sampled at FRAME_RATE. Uses the plot utilities in
    rotorpy.utils.camera_plotter.

    GIFs are named eval_env_<env>_spawn_<spawn>.gif in out_dir; returns the
    list of written paths.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from rotorpy.utils.camera_plotter import plot_world_with_camera

    cam_dt = 1.0 / fps
    sim_dt = 1.0 / SIM_RATE
    B = state['x'].shape[1]
    n_frames = int(np.ceil(t_final / cam_dt))

    # Render the camera frames once, batched across all running drones (the
    # same offline-render pattern as data collection), then build the GIFs.
    frames_img = []
    for k in range(n_frames):
        si = int(round(k * cam_dt / sim_dt))
        running = [b for b in range(B) if si < int(exit_timesteps[b])]
        if not running:
            break
        xq = {'x': torch.tensor(np.stack([state['x'][si, b] for b in running]),
                                dtype=torch.double, device=device),
              'q': torch.tensor(np.stack([state['q'][si, b] for b in running]),
                                dtype=torch.double, device=device)}
        img = camera.render([worlds[b] for b in running], xq)['image']
        img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        frames_img.append((si, running, img))

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for b in range(B):
        entries = [(si, img[j])
                   for (si, running, img) in frames_img
                   for j, r in enumerate(running) if r == b]
        if not entries:
            continue
        xs = np.stack([state['x'][si, b] for si, _ in entries])
        qs = np.stack([state['q'][si, b] for si, _ in entries])
        imgs = np.stack([im for _, im in entries])
        ts = [si * sim_dt for si, _ in entries]
        world_obj = worlds[b]
        e_abs = env_offset + int(env_idx[b])
        s = int(spawn_idx[b])

        fig = plt.figure(figsize=(11.0, 5.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
        ax3d = fig.add_subplot(gs[0], projection='3d')
        ax_img = fig.add_subplot(gs[1])

        def update(k):
            x, q = xs[k], qs[k]
            camera_pose = _camera_pose_from_state(camera.extrinsics, x, q)
            ax3d.clear()
            plot_world_with_camera(ax3d, world_obj, camera_pose=camera_pose,
                                   intrinsics=INTRINSICS, alpha=0.6,
                                   show_drone=True, drone_state={'x': x, 'q': q})
            ax3d.set_aspect('equal')
            ax3d.set_title('3D Scene  t=%.1f s' % ts[k])
            ax_img.clear()
            ax_img.imshow(imgs[k], origin='upper', interpolation='nearest')
            ax_img.set_title('Camera View  t=%.1f s' % ts[k])
            return []

        out_path = os.path.join(out_dir, "eval_env_%03d_spawn_%d.gif" % (e_abs, s))
        anim = FuncAnimation(fig, update, frames=len(xs), interval=1000.0 / fps)
        anim.save(out_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        paths.append(out_path)
        if verbose:
            print("  [animate] wrote %d-frame GIF of env %03d spawn %d -> %s"
                  % (len(xs), e_abs, s, os.path.abspath(out_path)))
    return paths

def eval_main(args):
    if not os.path.isfile(args.policy):
        sys.exit("policy checkpoint not found: %s (run 'train' first)"
                 % args.policy)
    device = _resolve_device(args.device)
    policy = build_policy()
    state_dict = torch.load(args.policy, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.to(device).eval()
    if args.verbose:
        print("Loaded policy from %s on %s"
              % (os.path.abspath(args.policy), device))

    rng = np.random.default_rng(args.seed)
    env_worlds, env_targets = build_envs(rng, args.num_envs, EVAL_ENV_OFFSET)
    worlds, starts, env_idx, spawn_idx = spawn_batch(
        rng, env_targets, args.spawns_per_env, env_worlds)
    B = len(worlds)
    targets_xy = env_targets[env_idx]
    targets = np.hstack([targets_xy, np.full((B, 1), HOVER_ALTITUDE)])

    if args.verbose:
        for e in range(args.num_envs):
            print("  [eval] env %03d: target (%.2f, %.2f)  %d spawn(s)"
                  % (EVAL_ENV_OFFSET + e, env_targets[e, 0], env_targets[e, 1],
                     args.spawns_per_env))
            for s in range(args.spawns_per_env):
                b = e * args.spawns_per_env + s
                print("    spawn %d: start (%.2f, %.2f, %.1f)"
                      % (s, starts[b, 0], starts[b, 1], starts[b, 2]))

    x0 = make_initial_states(starts, device)
    traj = BatchedHoverTraj(B, x0=targets, device=device)
    camera = make_camera(B, device)

    # Warm up each drone's history buffer with copies of its first frame.
    img0 = camera.render(worlds, {'x': x0['x'], 'q': x0['q']})['image']
    img0 = (img0.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    histories = [collections.deque([img0[b]] * HISTORY_LEN, maxlen=HISTORY_LEN)
                 for b in range(B)]
    controller = PolicyController(policy, camera, worlds, histories, device,
                                  verbose=args.verbose)

    batch_params = BatchedMultirotorParams([quad_params] * B, B, device)
    vehicle = BatchedMultirotor(batch_params, B, x0, device=device,
                                control_abstraction='cmd_vel', integrator='rk4')
    wind = make_wind(rng, B, device)
    imu = BatchedImu(B, device=device)
    print("Evaluating the policy on %d held-out drones (%d environments x %d "
          "spawns each) for %.1f s ..." % (B, args.num_envs,
                                           args.spawns_per_env, EVAL_T_FINAL))
    _, state, _, _, _, _, _, exit_timesteps, _ = simulate_batch(
        worlds, x0, vehicle, controller, traj, wind, imu,
        np.full(B, EVAL_T_FINAL), 1.0 / SIM_RATE,
        SAFETY_MARGIN, terminate=_never_exit, print_fps=PRINT_FPS)

    env_errs = {}
    for b in range(B):
        last = int(exit_timesteps[b]) - 1
        final_x = state['x'][last, b]
        err_horiz = float(np.linalg.norm(final_x[:2] - targets_xy[b]))
        err_vert = float(abs(final_x[2] - HOVER_ALTITUDE))
        e, s = int(env_idx[b]), int(spawn_idx[b])
        env_errs.setdefault(e, []).append((s, err_horiz, err_vert, final_x))
        print("  env %03d spawn %d  exited t=%.2f s  horiz_err %.2f m  "
              "vert_err %.2f m  final (%.2f, %.2f, %.2f)"
              % (EVAL_ENV_OFFSET + e, s, (last + 1) / SIM_RATE, err_horiz,
                 err_vert, final_x[0], final_x[1], final_x[2]))

    errs = np.array([v[1] for es in env_errs.values() for v in es])
    print("%d held-out spawns: mean horizontal error %.2f m (median %.2f, "
          "max %.2f)" % (len(errs), errs.mean(), np.median(errs), errs.max()))
    for e in range(args.num_envs):
        es = env_errs[e]
        print("  env %03d: mean horiz %.2f m over %d spawn(s)  (verts %s)"
              % (EVAL_ENV_OFFSET + e, np.mean([v[1] for v in es]), len(es),
                 ["%.2f" % v[2] for v in es]))
    print("Success (horiz < %.2f m and vertical in [%.1f,%.1f]): %d/%d"
          % (SUCCESS_THRESHOLD, HOVER_ALTITUDE - 0.3, HOVER_ALTITUDE + 0.3,
             int((errs < SUCCESS_THRESHOLD).sum()), len(errs)))

    if args.animate:
        animate_eval_runs(camera, worlds, env_idx, spawn_idx, state,
                          exit_timesteps, EVAL_T_FINAL, device,
                          args.animate_out, EVAL_ENV_OFFSET,
                          verbose=args.verbose)

""" 
CLI
"""

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda if available else cpu)')
    sub = parser.add_subparsers(dest='stage')

    p = sub.add_parser('generate', help='collect expert demonstrations')
    p.add_argument('--num-envs', type=int, default=NUM_ENVS,
                   help='number of independent environments (default: %d)'
                        % NUM_ENVS)
    p.add_argument('--spawns-per-env', type=int, default=SPAWNS_PER_ENV,
                   help='spawn points per environment; batch = num_envs x '
                        'spawns (default: %d)' % SPAWNS_PER_ENV)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--out', default=DATA_OUT,
                   help='directory to write episode npz files (default: %s)'
                        % DATA_OUT)
    p.add_argument('--verbose', action='store_true',
                   help='per-env targets/spawns and offline render progress')
    p.add_argument('--preview', action='store_true',
                   help='write a few preview frames of the first episode to '
                        'the output directory')

    p = sub.add_parser('train', help='train the policy on the generated data')
    p.add_argument('--dataset', default=DATA_OUT, 
                   help='directory of episode npz files (default: %s)'
                        % DATA_OUT)
    p.add_argument('--out', default=MODEL_OUT,
                   help='directory to write the trained policy (default: %s)'
                        % MODEL_OUT)
    p.add_argument('--epochs', type=int, default=EPOCHS,
                   help='number of training epochs (default: %d)' % EPOCHS)
    p.add_argument('--verbose', action='store_true',
                   help='dataset stats, per-batch losses, epoch timing')

    p = sub.add_parser('eval', help='closed-loop eval on held-out environments')
    p.add_argument('--num-envs', type=int, default=EVAL_NUM_ENVS,
                   help='number of held-out environments (default: %d)'
                        % EVAL_NUM_ENVS)
    p.add_argument('--spawns-per-env', type=int, default=SPAWNS_PER_ENV,
                   help='spawn points per environment (default: %d)'
                        % SPAWNS_PER_ENV)
    p.add_argument('--policy', default=os.path.join(MODEL_OUT, 'policy.pt'),
                   help='path to the trained policy (default: %s)'
                        % os.path.join(MODEL_OUT, 'policy.pt'))
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--verbose', action='store_true',
                   help='targets/spawns and per-second in-loop progress')
    p.add_argument('--animate', action='store_true',
                   help='write one GIF per eval drone (3D scene + camera view) '
                        'to --animate-out')
    p.add_argument('--animate-out', default=ANIMATE_OUT,
                   help='directory for the eval GIFs (default: %s)'
                        % ANIMATE_OUT)

    args = parser.parse_args()
    if args.stage is None:
        parser.print_help()
        return
    if args.stage == 'generate':
        generate_main(args)
    elif args.stage == 'train':
        train_main(args)
    elif args.stage == 'eval':
        eval_main(args)


if __name__ == "__main__":
    main()
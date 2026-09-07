"""
World visual-feature generators for RotorPy.

A visual feature is a 3D world position and an RGB color, optionally carrying a
generic descriptor vector (e.g. a classical or learned feature descriptor such
as SIFT or ALIKED). Generators place features on the blocks of a world (on
their surfaces and/or edges). The generator registry is the extension point: a
new placement strategy is added by subclassing ``WorldFeatureGenerator`` and
decorating the class with ``@register``, after which it is selectable by name
in a world's feature config. No other code needs to change.

Config schema (``world_data['features']`` dict, see ``World``):
    mode, generator name: 'regular', 'random', 'edge_uniform', 'edge_random'
    feature_spacing, grid spacing in meters (surface grid mode)
    feature_density, features per square meter of surface (random splatter
        mode; the per-face count is round(density * face_area))
    edge_spacing, uniform spacing along edges in meters (edge_uniform)
    edge_density, features per linear meter of edge (edge_random)
    descriptor_noise, standard deviation of Gaussian jitter added to each
        block's color to form per-feature RGB colors
    descriptor_dim, optional int; when set, each feature also receives a
        synthetic descriptor vector drawn i.i.d. Gaussian and L2-normalized to
        unit norm (useful for testing descriptor-based estimators)
    seed, optional int for reproducible generation; omit for fresh randomness
        per generation call

  Embedded features (bypass the generator entirely, no RNG involved):
    points, (N, 3) explicit feature positions
    colors, (N, 3) optional RGB colors; defaults to grey (0.6, 0.6, 0.6) if
        omitted
    descriptors, (N, D) optional generic descriptor vectors (any dimension D,
        e.g. 128-d SIFT or 96-d ALIKED); omitted features carry none
    descriptor_type, optional string label for the descriptor vector family
        (e.g. 'sift', 'aliked'); informational only
    blocks, (N,) optional int block index per feature; defaults to -1 (unknown)
"""

import numpy as np
import matplotlib.colors


_REGISTRY = {}


def register(name):
    """Decorator registering a generator class under ``name``."""

    def decorator(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def registered_mode_names():
    """Return the sorted list of registered generator mode names."""
    return sorted(_REGISTRY)


def _block_color(block):
    color = block.get('color', [0.6, 0.6, 0.6])
    if isinstance(color, str):
        color = matplotlib.colors.to_rgb(color)
    return np.asarray(color, dtype=np.float64)


def _block_faces(block):
    """Return the six faces of an axis-aligned block as tuples of
    ``(name, fixed_coord, u1, u2, v1, v2)`` where (u, v) parameterize the
    face plane and ``fixed_coord`` is the coordinate held constant."""
    xmin, xmax, ymin, ymax, zmin, zmax = block['extents']
    return [
        ('xmin', xmin, ymin, ymax, zmin, zmax),
        ('xmax', xmax, ymin, ymax, zmin, zmax),
        ('ymin', ymin, xmin, xmax, zmin, zmax),
        ('ymax', ymax, xmin, xmax, zmin, zmax),
        ('zmin', zmin, xmin, xmax, ymin, ymax),
        ('zmax', zmax, xmin, xmax, ymin, ymax),
    ]


def _block_edges(block):
    """Return the 12 edges of an axis-aligned block as tuples of
    ``(axis, fixed_a, fixed_b, lo, hi)`` where points along the edge are
    ``(fixed_a, fixed_b, t)`` permuted so that the varying coordinate
    ``axis`` runs from ``lo`` to ``hi``."""
    xmin, xmax, ymin, ymax, zmin, zmax = block['extents']
    edges = []
    for y in (ymin, ymax):
        for z in (zmin, zmax):
            edges.append(('x', y, z, xmin, xmax))
    for x in (xmin, xmax):
        for z in (zmin, zmax):
            edges.append(('y', x, z, ymin, ymax))
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            edges.append(('z', x, y, zmin, zmax))
    return edges


def _edge_points(axis, fixed_a, fixed_b, ts):
    """Build (n, 3) points for edge samples ``ts`` at given fixed coords."""
    n = ts.size
    ones = np.ones(n)
    if axis == 'x':
        return np.column_stack([ts, ones * fixed_a, ones * fixed_b])
    elif axis == 'y':
        return np.column_stack([ones * fixed_a, ts, ones * fixed_b])
    else:
        return np.column_stack([ones * fixed_a, ones * fixed_b, ts])


class WorldFeatureGenerator:
    """
    Base class for feature placement strategies.

    Subclasses define :meth:`generate`, which returns
    ``(features (N, 3), colors (N, 3), descriptors (N, D) or None, metadata)``
    where metadata is a list of dicts describing each feature's origin (e.g.
    'block_idx', 'surface' or 'edge'). Colors are RGB in [0, 1]; descriptors
    are an optional aligned matrix of generic D-dimensional vectors (with the
    same D for every feature), or None when no descriptor vectors are desired.
    A single RNG is created per generator so that each subclass draws a
    deterministic sequence for a given seed independent of any global RNG
    state.
    """

    name = None

    def __init__(self, blocks, descriptor_noise=0.0, seed=None, descriptor_dim=None, **kwargs):
        self.blocks = blocks
        self.descriptor_noise = descriptor_noise
        self.descriptor_dim = descriptor_dim
        self.rng = np.random.default_rng(seed)

    def colors_for(self, block, n):
        """Return ``(n, 3)`` RGB colors: the block color plus Gaussian jitter
        of standard deviation ``descriptor_noise``, clipped to [0, 1]."""
        base = _block_color(block)
        out = base + self.rng.normal(0.0, self.descriptor_noise, size=(n, 3))
        return np.clip(out, 0.0, 1.0)

    def descriptors_for(self, n):
        """Return ``(n, D)`` L2-normalized synthetic descriptor vectors if
        ``descriptor_dim`` is set, else None."""
        if not self.descriptor_dim:
            return None
        v = self.rng.normal(size=(n, self.descriptor_dim))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return v / norms

    def generate(self):
        """Place features. Returns (features, colors, descriptors, metadata)."""
        raise NotImplementedError

    def _stack(self, features, colors, descriptors, metadata):
        if not features:
            desc = (np.empty((0, self.descriptor_dim)) if self.descriptor_dim else None)
            return (np.empty((0, 3)), np.empty((0, 3)), desc, [])
        return (np.vstack(features), np.vstack(colors),
                np.vstack(descriptors) if descriptors else None, metadata)


@register('regular')
class GridFeatureGenerator(WorldFeatureGenerator):
    """
    Uniform grid of features on all block faces.

    Points are laid out on a regular grid with spacing ``feature_spacing``
    meters over each face (a face too small to fit a grid is skipped).
    """

    def __init__(self, blocks, feature_spacing=0.2, descriptor_noise=0.0, seed=None, descriptor_dim=None, **kwargs):
        super().__init__(blocks, descriptor_noise=descriptor_noise, seed=seed, descriptor_dim=descriptor_dim)
        self.spacing = feature_spacing

    def generate(self):
        features, colors, descriptors, metadata = [], [], [], []
        for block_idx, block in enumerate(self.blocks):
            for face_name, fixed, u1, u2, v1, v2 in _block_faces(block):
                u_range = np.arange(u1, u2 + self.spacing / 2, self.spacing)
                v_range = np.arange(v1, v2 + self.spacing / 2, self.spacing)
                if len(u_range) < 2 or len(v_range) < 2:
                    continue
                UU, VV = np.meshgrid(u_range, v_range)
                n = UU.size
                if face_name in ('xmin', 'xmax'):
                    points = np.column_stack([np.full(n, fixed), UU.ravel(), VV.ravel()])
                elif face_name in ('ymin', 'ymax'):
                    points = np.column_stack([UU.ravel(), np.full(n, fixed), VV.ravel()])
                else:
                    points = np.column_stack([UU.ravel(), VV.ravel(), np.full(n, fixed)])
                features.append(points)
                colors.append(self.colors_for(block, n))
                desc = self.descriptors_for(n)
                if desc is not None:
                    descriptors.append(desc)
                metadata.extend([{'block_idx': block_idx, 'surface': face_name}] * n)
        return self._stack(features, colors, descriptors, metadata)


@register('random')
@register('density')
class RandomDensityFeatureGenerator(WorldFeatureGenerator):
    """
    Random splatter on all block faces at a fixed areal density.

    The number of features on a face is ``round(feature_density * face_area)``,
    so a uniform spatial density is produced regardless of object size (large
    objects carry proportionally more features). Samples are inset 2% from the
    face boundaries to keep them off the shared edges.
    """

    def __init__(self, blocks, feature_density=50.0, descriptor_noise=0.0, seed=None, descriptor_dim=None, **kwargs):
        super().__init__(blocks, descriptor_noise=descriptor_noise, seed=seed, descriptor_dim=descriptor_dim)
        self.density = feature_density

    def generate(self):
        features, colors, descriptors, metadata = [], [], [], []
        for block_idx, block in enumerate(self.blocks):
            for face_name, fixed, u1, u2, v1, v2 in _block_faces(block):
                du, dv = u2 - u1, v2 - v1
                if du <= 0 or dv <= 0:
                    continue
                n = int(round(du * dv * self.density))
                if n <= 0:
                    continue
                inset_u, inset_v = 0.02 * du, 0.02 * dv
                us = self.rng.uniform(u1 + inset_u, u2 - inset_u, size=n)
                vs = self.rng.uniform(v1 + inset_v, v2 - inset_v, size=n)
                if face_name in ('xmin', 'xmax'):
                    points = np.column_stack([np.full(n, fixed), us, vs])
                elif face_name in ('ymin', 'ymax'):
                    points = np.column_stack([us, np.full(n, fixed), vs])
                else:
                    points = np.column_stack([us, vs, np.full(n, fixed)])
                features.append(points)
                colors.append(self.colors_for(block, n))
                desc = self.descriptors_for(n)
                if desc is not None:
                    descriptors.append(desc)
                metadata.extend([{'block_idx': block_idx, 'surface': face_name}] * n)
        return self._stack(features, colors, descriptors, metadata)


@register('edge_uniform')
class EdgeUniformFeatureGenerator(WorldFeatureGenerator):
    """
    Features spaced uniformly along all 12 block edges.

    Points are laid out with spacing ``edge_spacing`` meters along each edge,
    endpoints included (corners therefore belong to all three incident edges).
    """

    def __init__(self, blocks, edge_spacing=0.1, descriptor_noise=0.0, seed=None, descriptor_dim=None, **kwargs):
        super().__init__(blocks, descriptor_noise=descriptor_noise, seed=seed, descriptor_dim=descriptor_dim)
        self.spacing = edge_spacing

    def generate(self):
        features, colors, descriptors, metadata = [], [], [], []
        for block_idx, block in enumerate(self.blocks):
            for axis, fixed_a, fixed_b, lo, hi in _block_edges(block):
                length = hi - lo
                if length <= 0:
                    continue
                ts = np.arange(lo, hi + self.spacing / 2, self.spacing)
                n = ts.size
                if n == 0:
                    continue
                features.append(_edge_points(axis, fixed_a, fixed_b, ts))
                colors.append(self.colors_for(block, n))
                desc = self.descriptors_for(n)
                if desc is not None:
                    descriptors.append(desc)
                metadata.extend([{'block_idx': block_idx,
                                 'edge': (axis, fixed_a, fixed_b)}] * n)
        return self._stack(features, colors, descriptors, metadata)


@register('edge_random')
class EdgeRandomFeatureGenerator(WorldFeatureGenerator):
    """
    Random features along all 12 block edges at a fixed linear density.

    The number of features on an edge is ``round(edge_density * edge_length)``,
    uniformly distributed along it, giving a constant features-per-meter density
    on edges of any length.
    """

    def __init__(self, blocks, edge_density=50.0, descriptor_noise=0.0, seed=None, descriptor_dim=None, **kwargs):
        super().__init__(blocks, descriptor_noise=descriptor_noise, seed=seed, descriptor_dim=descriptor_dim)
        self.density = edge_density

    def generate(self):
        features, colors, descriptors, metadata = [], [], [], []
        for block_idx, block in enumerate(self.blocks):
            for axis, fixed_a, fixed_b, lo, hi in _block_edges(block):
                length = hi - lo
                if length <= 0:
                    continue
                n = int(round(length * self.density))
                if n <= 0:
                    continue
                ts = self.rng.uniform(lo, hi, size=n)
                features.append(_edge_points(axis, fixed_a, fixed_b, ts))
                colors.append(self.colors_for(block, n))
                desc = self.descriptors_for(n)
                if desc is not None:
                    descriptors.append(desc)
                metadata.extend([{'block_idx': block_idx,
                                 'edge': (axis, fixed_a, fixed_b)}] * n)
        return self._stack(features, colors, descriptors, metadata)


def generate_features(blocks, config):
    """Dispatch to the generator named by ``config['mode']``."""
    mode = config['mode']
    if mode not in _REGISTRY:
        raise ValueError("Unknown feature mode {!r}; available: {}".format(mode, registered_mode_names()))
    params = {k: v for k, v in config.items() if k != 'mode'}
    generator = _REGISTRY[mode](blocks, **params)
    return generator.generate()


def features_from_config(blocks, config):
    """
    Resolve a feature config dict into concrete features.

    If the config embeds explicit 'points', those are used verbatim (no
    generator, no RNG); otherwise the generator named by ``config['mode']``
    places them.

    Inputs:
        blocks, list of block dicts from world_data
        config, dict per the module docstring

    Outputs:
        features, (N, 3) float array
        colors, (N, 3) float array of RGB colors in [0, 1]
        descriptors, (N, D) float array or None when no descriptor vectors
        metadata, list of dicts with a 'block_idx' key per feature
    """
    points = config.get('points')
    if points is None:
        return generate_features(blocks, config)

    features = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    colors = config.get('colors')
    if colors is None:
        colors = np.full(features.shape, 0.6, dtype=np.float64)
    else:
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if colors.shape[0] != features.shape[0]:
            raise ValueError("embedded features: 'colors' count {} does not match "
                             "'points' count {}".format(colors.shape[0], features.shape[0]))

    descriptors = config.get('descriptors')
    if descriptors is not None:
        descriptors = np.asarray(descriptors, dtype=np.float64)
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(-1, 1)
        if descriptors.shape[0] != features.shape[0]:
            raise ValueError("embedded features: 'descriptors' count {} does not match "
                             "'points' count {}".format(descriptors.shape[0], features.shape[0]))

    blocks_entry = config.get('blocks')
    if blocks_entry is None:
        metadata = [{'block_idx': -1} for _ in range(features.shape[0])]
    else:
        indices = np.asarray(blocks_entry).ravel()
        if indices.shape[0] != features.shape[0]:
            raise ValueError("embedded features: 'blocks' count {} does not match "
                             "'points' count {}".format(indices.shape[0], features.shape[0]))
        metadata = [{'block_idx': int(i)} for i in indices]
    return features, colors, descriptors, metadata
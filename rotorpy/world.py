import json
import numpy as np

from rotorpy.utils.shapes import Cuboid
from rotorpy.utils.numpy_encoding import NumpyJSONEncoder, to_ndarray
from rotorpy.world_features import features_from_config

def interp_path(path, res):
    if path.size == 3:
        # There's only one datapoint. Return the point. 
        return path.reshape(1,-1)
    else:
        cumdist = np.cumsum(np.linalg.norm(np.diff(path, axis=0),axis=1))
        if cumdist[-1] > 0:
            t = np.insert(cumdist,0,0)
            ts = np.arange(0, cumdist[-1], res)
            pts = np.empty((ts.size, 3), dtype=np.float64)
            for k in range(3):
                pts[:,k] = np.interp(ts, t, path[:,k])
        else:
            pts = path[[0],:]
        return pts

class World(object):

    def __init__(self, world_data, add_features=False, feature_mode='regular', feature_spacing=0.2,
                 N_features_per_surface=100, seed=None, descriptor_noise=0.05,
                 feature_density=50.0, edge_spacing=0.1, edge_density=50.0,
                 descriptor_dim=None):
        """
        Construct World object from data. Instead of using this constructor
        directly, see also class methods 'World.from_file()' for building a
        world from a saved .json file or 'World.grid_forest()' for building a
        world object of a parameterized style.

        Parameters:
            world_data, dict containing keys 'bounds' and 'blocks'
                bounds, dict containing key 'extents'
                    extents, list of [xmin, xmax, ymin, ymax, zmin, zmax]
                blocks, list of dicts containing keys 'extents' and 'color'
                    extents, list of [xmin, xmax, ymin, ymax, zmin, zmax]
                    color, color specification
                features, optional dict describing how to place visual features
                    (see rotorpy.world_features for the full schema). If present
                    it enables feature generation regardless of add_features.
                    It may either configure a generator:
                        mode, 'regular', 'random' (areal-density splatter),
                            'edge_uniform', or 'edge_random'
                        feature_spacing, grid spacing, m ('regular')
                        feature_density, features per m^2 ('random')
                        edge_spacing, edge spacing, m ('edge_uniform')
                        edge_density, features per m of edge ('edge_random')
                        descriptor_noise, RGB jitter standard deviation
                        seed, optional int for reproducibility
                    or embed the concrete features directly, bypassing any
                    generator and RNG entirely (exact across trials and
                    reloads):
                        points, (N, 3) feature positions
                        colors, (N, 3) RGB colors (optional)
                        descriptors, (N, D) generic descriptor vectors, any
                            dimension D (optional; e.g. 128-d SIFT)
                        descriptor_type, string label for the 'descriptors'
                            family (optional, informational)
                        blocks, (N,) per-feature block indices (optional)
            add_features, if True, generates surface features on all block faces.
                Ignored if world_data contains a 'features' section.
            feature_mode, feature placement mode: 'regular' (uniform grid),
                'random' (random splatter at areal density), 'edge_uniform',
                or 'edge_random'. Only used when world_data has no 'features'.
            feature_spacing, spacing between grid points, m (only for 'regular' mode)
            N_features_per_surface, DEPRECATED and ignored. 'random' mode now
                uses feature_density (features per m^2) so that feature density
                is uniform regardless of object size.
            seed, random seed for reproducibility (only for random feature generation)
            descriptor_noise, standard deviation of Gaussian jitter added to each block's
                color to form per-feature RGB colors
            feature_density, features per square meter (only for 'random' mode)
            edge_spacing, spacing along edges, m (only for 'edge_uniform' mode)
            edge_density, features per meter of edge (only for 'edge_random' mode)
            descriptor_dim, if set, each generated feature also carries a
                synthetic L2-normalized descriptor vector of this dimension
        """
        self.world = world_data
        self.add_features = add_features
        self.feature_mode = feature_mode
        self.feature_spacing = feature_spacing
        self.N_features_per_surface = N_features_per_surface
        self.feature_density = feature_density
        self.edge_spacing = edge_spacing
        self.edge_density = edge_density
        self.seed = seed
        self.descriptor_noise = descriptor_noise
        self.descriptor_dim = descriptor_dim

        # Resolve the feature config: an embedded world_data['features'] section
        # takes precedence over the constructor arguments.
        if 'features' in world_data:
            self.features_config = dict(world_data['features'])
        elif add_features:
            self.features_config = {'mode': feature_mode, 'feature_spacing': feature_spacing,
                                    'feature_density': feature_density,
                                    'edge_spacing': edge_spacing,
                                    'edge_density': edge_density,
                                    'descriptor_noise': descriptor_noise,
                                    'seed': seed}
            if descriptor_dim is not None:
                self.features_config['descriptor_dim'] = descriptor_dim
        else:
            self.features_config = None

        # Generate surface features if requested.
        self._surface_features = None
        self._feature_blocks = None
        self._feature_metadata = None
        self._feature_colors = None
        self._feature_descriptors = None
        self._feature_descriptor_type = None
        if self.features_config is not None:
            self.generate_surface_features(config=self.features_config)

    @classmethod
    def from_file(cls, filename):
        """
        Read world definition from a .json text file and return World object.

        Parameters:
            filename

        Returns:
            world, World object

        Example use:
            my_world = World.from_file('my_filename.json')
        """
        with open(filename) as file:
            return cls(to_ndarray(json.load(file)))

    def to_file(self, filename, include_features=False):
        """
        Write world definition to a .json text file.

        Parameters:
            filename
            include_features, if True, embed the currently generated features
                into the file's 'features' section (points, colors, optional
                descriptor vectors and their type, and per-feature block
                indices), so loading the file reproduces the exact same features
                without relying on a random seed. If False, only the generator
                configuration is written (reproducibility then depends on its
                seed).

        Example use:
            my_world.to_file('my_filename.json')
        """
        world_data = self.world
        if self.features_config is not None:
            world_data = dict(self.world)
            features = dict(self.features_config)
            if include_features and self._surface_features is not None:
                features['points'] = np.asarray(self._surface_features).tolist()
                features['colors'] = (np.asarray(self._feature_colors).tolist()
                                      if self._feature_colors is not None
                                      else [])
                if self._feature_descriptors is not None:
                    features['descriptors'] = np.asarray(self._feature_descriptors).tolist()
                if self._feature_descriptor_type is not None:
                    features['descriptor_type'] = self._feature_descriptor_type
                if self._feature_blocks is not None:
                    features['blocks'] = np.asarray(self._feature_blocks, dtype=int).tolist()
            world_data['features'] = features
        with open(filename, 'w') as file:  # TODO check for directory to exist
            file.write(json.dumps(world_data, cls=NumpyJSONEncoder, indent=4))

    def closest_points(self, points):
        """
        For each point, return the closest occupied point in the world and the
        distance to that point. This is appropriate for computing sphere-vs-world
        collisions.

        Input
            points, (N,3)
        Returns
            closest_points, (N,3)
            closest_distances, (N,)
        """

        closest_points = np.empty_like(points)
        closest_distances = np.full(points.shape[0], np.inf)
        p = np.empty_like(points)
        for block in self.world.get('blocks', []):
            # Computation takes advantage of axes-aligned blocks. Note that
            # scipy.spatial.Rectangle can compute this distance, but wouldn't
            # return the point itself.
            r = block['extents']
            for i in range(3):
                p[:, i] = np.clip(points[:, i], r[2*i], r[2*i+1])
            d = np.linalg.norm(points-p, axis=1)
            mask = d < closest_distances
            closest_points[mask, :] = p[mask, :]
            closest_distances[mask] = d[mask]
        return (closest_points, closest_distances)

    def generate_surface_features(self, mode=None, spacing=None, N_features_per_surface=None, seed=None,
                                  descriptor_noise=None, feature_density=None, edge_spacing=None,
                                  edge_density=None, descriptor_dim=None, config=None):
        """
        Generate feature points on all exposed block surfaces and/or edges.

        The placement strategy is resolved from ``config`` if given, otherwise
        from the explicit keyword arguments (see ``World.__init__`` for the
        meaning of each parameter). ``config`` may alternately embed explicit
        'points'/'colors'/'descriptors' to load features verbatim without
        generating.

        Parameters:
            config, optional dict describing feature generation, see the
                'features' key documented in World.__init__
            mode, 'regular', 'random', 'edge_uniform', or 'edge_random'
            spacing, grid or edge spacing, m (modes 'regular'/'edge_uniform')
            feature_density, features per m^2 (mode 'random')
            edge_density, features per m of edge (mode 'edge_random')
            seed, random seed for reproducibility
            descriptor_noise, standard deviation of Gaussian jitter added to
                each block's color to form per-feature RGB colors
            descriptor_dim, if set, each generated feature also carries a
                synthetic L2-normalized descriptor vector of this dimension
            N_features_per_surface, DEPRECATED and ignored

        Returns:
            features: (N, 3) array of world coordinate features
            features_metadata: list of dicts with at least a 'block_idx' key;
                surface features also carry a 'surface' key and edge features
                an 'edge' key
        """
        if config is None:
            config = {'mode': mode if mode is not None else 'regular'}
            if spacing is not None:
                config['feature_spacing'] = spacing
            if feature_density is not None:
                config['feature_density'] = feature_density
            if edge_spacing is not None:
                config['edge_spacing'] = edge_spacing
            if edge_density is not None:
                config['edge_density'] = edge_density
            if seed is not None:
                config['seed'] = seed
            if descriptor_noise is not None:
                config['descriptor_noise'] = descriptor_noise
            if descriptor_dim is not None:
                config['descriptor_dim'] = descriptor_dim

        features, colors, descriptors, metadata = features_from_config(self.world.get('blocks', []), config)

        self._surface_features = features
        self._feature_colors = colors
        self._feature_descriptors = descriptors
        self._feature_descriptor_type = config.get('descriptor_type')
        self._feature_metadata = metadata
        self._feature_blocks = [m['block_idx'] for m in metadata]
        self.features_config = dict(config)

        return features, metadata

    def min_dist_boundary(self, points):
        """
        For each point, calculate the minimum distance to the boundary checking, x,y,z. A negative distance means the
        point is outside the boundary
        Input
            points, (N,3)
        Returns
            closest_distances, (N,)
        """

        # Bounds with upper limits negated [xmin, -xmax, ymin, -ymax, ...]
        test_bounds = np.array(self.world['bounds']['extents'])
        test_bounds[1::2] = -test_bounds[1::2]

        # Repeated coordinates with second entry negated [x, -x, y, -y, ...]
        test_points = np.repeat(points, 2, 1)
        test_points[:,1::2] = -test_points[:,::2]

        # Compute [x-xmin, xmax-x, y-ymin, ymax-y, z-zmin, zmax-z].
        # Minimum distance is the minimum for each point to all walls.
        distances = test_points - test_bounds
        min_distances = np.amin(distances, 1)

        return min_distances

    def path_collisions(self, path, margin):
        """
        Densely sample the path and check for collisions. Return a boolean mask
        over the samples and the sample points themselves.
        """
        pts = interp_path(path, res=0.001)
        collisions = self.collisions(pts, margin)
        return pts[collisions]

    def collisions(self, points, margin):
        """
        Return a boolean mask over ``points`` marking which are in collision
        with the world (within ``margin`` of a block, or outside the world
        boundary).

        Unlike ``closest_points`` (which iterates over blocks in Python), the
        block distance computation is vectorized over both points and blocks,
        so it is cheap to call on a whole batch of vehicle positions at every
        simulation step.

        Inputs:
            points, (N, 3) array of world positions
            margin, the radius of the ball surrounding each point to determine
                if a collision occurs, m

        Returns:
            collisions, (N,) bool array, True where the point collides
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        blocks = self.world.get('blocks', [])
        if len(blocks) == 0:
            closest_dist = np.full(pts.shape[0], np.inf)
        else:
            # Clip each point to each axis-aligned block and take the minimum
            # distance, all vectorized over (points, blocks).
            r = np.array([b['extents'] for b in blocks], dtype=np.float64).reshape(-1, 6)
            clipped = np.clip(pts[:, None, :],
                              r[None, :, 0::2], r[None, :, 1::2])      # (N, M, 3)
            dist = np.linalg.norm(pts[:, None, :] - clipped, axis=-1)  # (N, M)
            closest_dist = dist.min(axis=-1)

        collisions_blocks = closest_dist < margin
        collisions_points = self.min_dist_boundary(pts) < 0
        return np.logical_or(collisions_points, collisions_blocks)

    def draw_empty_world(self, ax):
        """
        Draw just the world without any obstacles yet. The boundary is represented with a black line.
        Parameters:
            ax, Axes3D object
        """
        (xmin, xmax, ymin, ymax, zmin, zmax) = self.world['bounds']['extents']

        # Set axes limits all equal to approximate 'axis equal' display.
        x_width = xmax-xmin
        y_width = ymax-ymin
        z_width = zmax-zmin
        width = np.max((x_width, y_width, z_width))
        ax.set_xlim((xmin, xmin+width))
        ax.set_ylim((ymin, ymin+width))
        ax.set_zlim((zmin, zmin+width))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        c = Cuboid(ax, xmax - xmin, ymax - ymin, zmax - zmin, alpha=0.01, linewidth=1, edgecolors='k')
        c.transform(position=(xmin, ymin, zmin))
        return list(c.artists)

    def draw(self, ax, alpha=None, edgecolor=None, facecolor=None):
        """
        Draw world onto existing Axes3D axes and return artists corresponding to the
        blocks.

        Parameters:
            ax, Axes3D object

        Returns:
            block_artists, list of Artists associated with blocks

        Example use:
            my_world.draw(ax)
        """
        bounds_artists = self.draw_empty_world(ax)

        if alpha is None:
            alpha = 0.7
        
        if edgecolor is None:
            edgecolor = 'k'

        block_artists = []
        for b in self.world.get('blocks', []):
            (xmin, xmax, ymin, ymax, zmin, zmax) = b['extents']
            if facecolor is None:
                fc = b.get('color', None)
            else:
                fc = facecolor
            c = Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=alpha, linewidth=1, edgecolors=edgecolor, facecolors=fc)
            c.transform(position=(xmin, ymin, zmin))
            block_artists.extend(c.artists)
        return bounds_artists + block_artists

    def draw_line(self, ax, points, color=None, linewidth=2):
        path_length = np.sum(np.linalg.norm(np.diff(points, axis=0),axis=1))
        pts = interp_path(points, res=path_length/1000)
        # The scatter object is assigned a single z-order value. Split for better occlusion rendering.
        for p in np.array_split(pts, 20):
            ax.scatter(p[:,0], p[:,1], p[:,2], s=linewidth**2, c=color, edgecolors='none', depthshade=False)

    def draw_points(self, ax, points, color=None, markersize=4):
        # The scatter object is assigned a single z-order value. Split for better occlusion rendering.
        for p in np.array_split(points, 20):
            ax.scatter(p[:,0], p[:,1], p[:,2], s=markersize**2, c=color, edgecolors='none', depthshade=False)

    def get_block_bounding_boxes(self):
        """
        Get axis-aligned bounding boxes for all blocks.

        Returns:
            boxes: list of (xmin, xmax, ymin, ymax, zmin, zmax) tuples for each block
        """
        boxes = []
        for block in self.world.get('blocks', []):
            boxes.append(block['extents'])
        return boxes

    def get_surface_features(self):
        """Return the cached surface features if they exist."""
        return self._surface_features

    def get_feature_colors(self):
        """Return the (N, 3) RGB colors of the surface features, if they exist."""
        return self._feature_colors

    def get_feature_descriptors(self):
        """Return the (N, D) generic descriptor vectors of the surface features,
        if they exist, else None. The dimension D is set by the generator or
        embedded data (e.g. 128-d SIFT, 96-d ALIKED)."""
        return self._feature_descriptors

    def get_feature_descriptor_type(self):
        """Return the descriptor vector family label (e.g. 'sift'), or None."""
        return self._feature_descriptor_type

    # The follow class methods are convenience functions for building different
    # kinds of parametric worlds.

    @staticmethod
    def _feature_kwargs(surface_kwargs):
        """Map feature-generation keyword arguments (see World.__init__) to the
        constructor arguments they feed."""
        defaults = {
            'feature_mode': 'regular',
            'feature_spacing': 0.2,
            'N_features_per_surface': None,
            'seed': None,
            'descriptor_noise': 0.05,
            'feature_density': 50.0,
            'edge_spacing': 0.1,
            'edge_density': 50.0,
            'descriptor_dim': None,
        }
        return {key: surface_kwargs.get(key, default) for key, default in defaults.items()}

    @classmethod
    def empty(cls, extents, add_features=False, **surface_kwargs):
        """
        Return World object for bounded empty space.

        Parameters:
            extents, tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
            add_features, if True, generates surface features
            surface_kwargs, keyword arguments for surface feature generation (see generate_surface_features)

        Returns:
            world, World object

        Example use:
            my_world = World.empty((xmin, xmax, ymin, ymax, zmin, zmax))
        """
        bounds = {'extents': extents}
        blocks = []
        world_data = {'bounds': bounds, 'blocks': blocks}
        return cls(world_data, add_features=add_features, **World._feature_kwargs(surface_kwargs))

    @classmethod
    def grid_forest(cls, n_rows, n_cols, width, height, spacing, add_features=False, **surface_kwargs):
        """
        Return World object describing a grid forest world parameterized by
        arguments. The boundary extents fit tightly to the included trees.

        Parameters:
            n_rows, rows of trees stacked in the y-direction
            n_cols, columns of trees stacked in the x-direction
            width, weight of square cross section trees
            height, height of trees
            spacing, spacing between centers of rows and columns
            add_features, if True, generates surface features
            surface_kwargs, keyword arguments for surface feature generation (see generate_surface_features)

        Returns:
            world, World object

        Example use:
            my_world = World.grid_forest(n_rows=4, n_cols=3, width=0.5, height=3.0, spacing=2.0)
        """

        # Bounds are outer boundary for world, which are an implicit obstacle.
        x_max = (n_cols-1)*spacing + width
        y_max = (n_rows-1)*spacing + width
        bounds = {'extents': [0, x_max, 0, y_max, 0, height]}

        # Blocks are obstacles in the environment.
        x_root = spacing * np.arange(n_cols)
        y_root = spacing * np.arange(n_rows)
        blocks = []
        for x in x_root:
            for y in y_root:
                blocks.append({'extents': [x, x+width, y, y+width, 0, height], 'color': [1, 0, 0]})

        world_data = {'bounds': bounds, 'blocks': blocks}
        return cls(world_data, add_features=add_features, **World._feature_kwargs(surface_kwargs))

    @classmethod
    def random_forest(cls, world_dims, tree_width, tree_height, num_trees, add_features=False, **surface_kwargs):
        """
        Return World object describing a random forest world parameterized by
        arguments.

        Parameters:
            world_dims, a tuple of (xmax, ymax, zmax). xmin,ymin, and zmin are set to 0.
            tree_width, weight of square cross section trees
            tree_height, height of trees
            num_trees, number of trees
            add_features, if True, generates surface features
            surface_kwargs, keyword arguments for surface feature generation (see generate_surface_features)

        Returns:
            world, World object
        """

        if 'seed' in surface_kwargs:
            np.random.seed(surface_kwargs['seed'])

        # Bounds are outer boundary for world, which are an implicit obstacle.
        bounds = {'extents': [0, world_dims[0], 0, world_dims[1], 0, world_dims[2]]}

        # Blocks are obstacles in the environment.
        xs = np.random.uniform(0, world_dims[0], num_trees)
        ys = np.random.uniform(0, world_dims[1], num_trees)
        pts = np.stack((xs, ys), axis=-1) # min corner location of trees
        w, h = tree_width, tree_height
        blocks = []
        for pt in pts:
            extents = list(np.round([pt[0], pt[0]+w, pt[1], pt[1]+w, 0, h], 2))
            blocks.append({'extents': extents, 'color': [1, 0, 0]})

        world_data = {'bounds': bounds, 'blocks': blocks}
        return cls(world_data, add_features=add_features, feature_mode=surface_kwargs.get('feature_mode', 'regular'), feature_spacing=surface_kwargs.get('feature_spacing', 0.2), N_features_per_surface=surface_kwargs.get('N_features_per_surface', 100), seed=surface_kwargs.get('seed', None), descriptor_noise=surface_kwargs.get('descriptor_noise', 0.05), descriptor_dim=surface_kwargs.get('descriptor_dim', None))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Display a map file in a Matplotlib window.')
    parser.add_argument('filename', help="Filename for map file json.")
    p = parser.parse_args()

    file = Path(p.filename)
    world = World.from_file(file)

    fig = plt.figure(f"{file.name}")
    ax = fig.add_subplot(projection='3d')
    world.draw(ax)

    plt.show()

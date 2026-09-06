import numpy as np
from scipy.spatial.transform import Rotation

try:
    import torch
except ImportError:
    torch = None


# Sentinel distinguishing "no noise_params keyword passed" from an explicit
# noise_params=None (which disables the effect for that call).
_NOISE_UNSET = object()

# Which per-feature data a render/measurement returns. 'all' returns both the
# RGB colors and the descriptor vectors. 'rgb' returns only colors (descriptors
# are set to None), and 'descriptors' returns only the descriptor vectors
# (colors are set to None). The rendered image is unchanged: colors are always
# used for splatting regardless of this flag. The savings matter when frames
# are collected in bulk -- descriptors are the bulk of the memory (K, N, D)
# and are irrelevant to image-only or RGB-based learning, while colors are
# redundant for VO/VIO pipelines that consume descriptors only.
_FEATURE_OUTPUTS = ('all', 'rgb', 'descriptors')


def _resolve_feature_output(requested, instance_default):
    """
    Resolve a feature_output selection, validating the value.

    Inputs:
        requested, the per-call override, or None to fall back to the camera's
            instance default
        instance_default, self.feature_output from the constructor

    Outputs:
        feature_output, one of 'all', 'rgb', 'descriptors'

    Raises:
        ValueError on an unrecognized value.
    """
    if requested is None:
        requested = instance_default
    if requested not in _FEATURE_OUTPUTS:
        raise ValueError("feature_output must be one of {}, got {!r}".format(_FEATURE_OUTPUTS, requested))
    return requested


def _coerce_noise_params(noise_params, default_splat_radius):
    """
    Validate a noise_params dict and fill in defaults.

    Inputs:
        noise_params, None to disable noise, or a dict with keys
            feature_rate, mean number of injected features per frame (required)
            splat_radius, pixel radius of each injected feature (defaults to
                default_splat_radius)
            intensity, color strength in [0, 1] (default 1.0)
            seed, optional int for reproducible injection
        default_splat_radius, splat radius used when one is not given

    Outputs:
        coerced, a normalized dict, or None if noise_params is None

    Raises:
        ValueError if noise_params is malformed.
    """
    if noise_params is None:
        return None
    if not isinstance(noise_params, dict):
        raise ValueError("noise_params must be a dict or None, got {}".format(type(noise_params).__name__))
    params = dict(noise_params)

    if 'feature_rate' not in params:
        raise ValueError("noise_params must specify 'feature_rate'")
    rate = float(params['feature_rate'])
    if rate < 0:
        raise ValueError("noise_params['feature_rate'] must be >= 0, got {}".format(rate))
    params['feature_rate'] = rate

    radius = params.get('splat_radius', default_splat_radius)
    if radius is None:
        radius = default_splat_radius
    radius = int(radius)
    if radius < 0:
        raise ValueError("noise_params['splat_radius'] must be >= 0, got {}".format(radius))
    params['splat_radius'] = radius

    intensity = float(params.get('intensity', 1.0))
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("noise_params['intensity'] must be in [0, 1], got {}".format(intensity))
    params['intensity'] = intensity

    return params


class PinholeCamera:
    """
    Simulated pinhole camera sensor.

    Projects world-surface features onto an image plane using an OpenCV-style
    pinhole + radial/tangential distortion model, and explicitly handles
    occlusion by the axis-aligned blocks of a world.

    Camera frame convention: the camera looks along +z, with x to the right
    and y down (OpenCV-style). World-to-camera transform is given by
        p_c = R_WC @ (p_W - p_WC),   R_WC = R_BC @ R_WB^T,
    where R_WB is the vehicle's body-to-world rotation (from the state
    quaternion) and R_BC is the body-to-camera rotation from the extrinsics.

    Projection model (OpenCV, with distortion coefficients [k1, k2, p1, p2, k3]):
        x, y        normalized coordinates from p_c (x = X/Z, y = Y/Z)
        r2          = x^2 + y^2
        x'          = x*(1 + k1*r2 + k2*r2^2 + k3*r2^3) + 2*p1*x*y + p2*(r2 + 2*x^2)
        y'          = y*(1 + k1*r2 + k2*r2^2 + k3*r2^3) + p1*(r2 + 2*y^2) + 2*p2*x*y
        u, v        = fx*x' + cx, fy*y' + cy

    Parameters:
        intrinsics, dict with keys fx, fy, width, height, cx, cy, dist_coeffs
            (see the default below). dist_coeffs is [k1, k2, p1, p2, k3].
        extrinsics, dict with keys position (3,) and orientation (4,) quaternion
            [i, j, k, w]. The quaternion is the rotation from the body frame to
            the camera frame.
        near_plane, minimum camera-frame z (in meters) for a feature to be
            rendered.
        frame_rate, the rate at which the simulator captures frames from this
            camera, Hz. None (default) renders at every simulation step. This
            only affects collection during simulate(); direct calls to
            measurement()/render() are unaffected.
        splat_radius, default pixel half-width of the square patch each feature
            is splatted onto during render(). Can be overridden per call.
        noise_params, optional dict enabling a visual noise effect applied by
            measurement() on top of render(). See __init__ for the tuning knobs.
        feature_output, which per-feature data measurement()/render() return:
            'all' (default) for both RGB colors and descriptor vectors, 'rgb'
            for colors only, or 'descriptors' for descriptor vectors only. See
            __init__ for details.

    State space:
        The vehicle state dict is expected to contain 'x' (3,) position and
        'q' (4,) orientation quaternion [i, j, k, w].
    """
    def __init__(self, intrinsics=None, extrinsics=None, near_plane=0.05, frame_rate=None,
                 splat_radius=1, noise_params=None, feature_output='all'):
        """
        Parameters:
            intrinsics, dict of camera intrinsics, see class docstring
            extrinsics, dict of camera extrinsics, see class docstring
            near_plane, minimum camera-frame z for a feature to be rendered, m
            frame_rate, frame capture rate used by simulate(), Hz; None renders
                at every simulation step
            splat_radius, default splat radius used by render(), pixels
            noise_params, None (default) for no noise, or a dict of tuning
                knobs for the measurement() visual noise effect which randomly
                injects synthetic features onto each frame:
                    feature_rate, mean number of injected features per frame;
                        the per-frame count is drawn from Poisson(feature_rate)
                    splat_radius, pixel patch size of each injected feature;
                        defaults to the camera's splat_radius
                    intensity, color strength in [0, 1]; 1.0 is full-strength
                        random colors, lower values give subtler artifacts
                    seed, optional int making the injection reproducible for a
                        given call; omit for fresh randomness each frame
                Injected features carry no 3D position, so only the rendered
                image is modified; keypoint/depth/visibility outputs stay clean.
            feature_output, which per-feature data render()/measurement() put
                in their output dicts. This trades memory for completeness when
                bulk-collecting frames:
                    'all' (default), return both the RGB colors and the
                        descriptor vectors (colors, visible_colors, descriptors,
                        visible_descriptors)
                    'rgb', return only the RGB colors (descriptors and
                        visible_descriptors are None). Use for image/RGB-based
                        learning or when descriptors are not needed.
                    'descriptors', return only the descriptor vectors (colors
                        and visible_colors are None). Use for VO/VIO pipelines
                        that consume descriptors only.
                The rendered image is always splatted using colors regardless
                of this setting. Can be overridden per call through render()/
                measurement().
        """
        if frame_rate is not None and frame_rate <= 0:
            raise ValueError("frame_rate must be positive or None, got {}".format(frame_rate))
        if int(splat_radius) < 0:
            raise ValueError("splat_radius must be non-negative, got {}".format(splat_radius))

        self.frame_rate = frame_rate
        self.splat_radius = int(splat_radius)
        self.feature_output = _resolve_feature_output(feature_output, 'all')
        if intrinsics is None:
            intrinsics = {'fx': 500.0, 'fy': 500.0, 'width': 640, 'height': 480, 'cx': 320.0, 'cy': 240.0,
                          'dist_coeffs': np.array([0.0, 0.0, 0.0, 0.0, 0.0])}  # [k1, k2, p1, p2, k3]
        if extrinsics is None:
            extrinsics = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.0, 0.0, 0.0, 1.0])}

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.near_plane = near_plane
        self.noise_params = _coerce_noise_params(noise_params, default_splat_radius=self.splat_radius)

        # Denormalized intrinsics.
        self.fx = float(intrinsics['fx'])
        self.fy = float(intrinsics['fy'])
        self.width = int(intrinsics['width'])
        self.height = int(intrinsics['height'])
        self.cx = float(intrinsics['cx'])
        self.cy = float(intrinsics['cy'])
        self.dist_coeffs = np.asarray(intrinsics['dist_coeffs'], dtype=np.float64)

        # Extrinsics: body-to-camera position and rotation.
        p_BC = np.asarray(extrinsics['position'], dtype=np.float64)
        q_BC = np.asarray(extrinsics['orientation'], dtype=np.float64)
        if p_BC.shape != (3,):
            raise ValueError("extrinsics['position'] must have shape (3,), got {}".format(p_BC.shape))
        if q_BC.shape != (4,):
            raise ValueError("extrinsics['orientation'] must have shape (4,), got {}".format(q_BC.shape))
        if self.dist_coeffs.shape != (5,):
            raise ValueError("intrinsics['dist_coeffs'] must have shape (5,), got {}".format(self.dist_coeffs.shape))
        self.p_BC = p_BC
        self.R_BC = Rotation.from_quat(q_BC).as_matrix()  # body -> camera

    def compute_camera_pose(self, state):
        """
        Compute the camera's world pose from the vehicle state.

        Inputs:
            state, a dict describing the vehicle state with keys
                x, position, m, shape=(3,)
                q, orientation quaternion [i, j, k, w], shape=(4,)

        Outputs:
            camera_pose, dict with keys
                x, camera world position, shape=(3,)
                q, world-to-camera quaternion [i, j, k, w], shape=(4,)
        """
        x_WB = np.asarray(state['x'], dtype=np.float64)
        q_WB = np.asarray(state['q'], dtype=np.float64)
        if x_WB.shape != (3,):
            raise ValueError("state['x'] must have shape (3,), got {}".format(x_WB.shape))
        if q_WB.shape != (4,):
            raise ValueError("state['q'] must have shape (4,), got {}".format(q_WB.shape))

        R_WB = Rotation.from_quat(q_WB).as_matrix()  # body -> world

        # Camera world position: p_WC = state['x'] + R_WB @ p_BC.
        p_WC = x_WB + R_WB @ self.p_BC

        # World-to-camera quaternion: q_WC = q_BC * q_WB^{-1}.
        # R_WC = R_BC @ R_WB^T  (body-to-camera composed with world-to-body).
        q_WC = (Rotation.from_quat(np.asarray(self.extrinsics['orientation'], dtype=np.float64)) * Rotation.from_quat(q_WB).inv()).as_quat()

        return {'x': p_WC, 'q': q_WC}

    def measurement(self, state, world, **render_kwargs):
        """
        Compute a camera measurement given the vehicle state and world.

        This is the primary sensor interface, consistent with the measurement()
        pattern used by other RotorPy sensors (IMU, mocap, range sensors). The
        raw render() output is produced first, then optional measurement
        effects are layered on top. The visual noise effect (see noise_params
        in __init__) randomly injects synthetic features onto the frame when
        enabled; additional effects can be added here later.

        Inputs:
            state, a dict describing the vehicle state with keys
                x, position, m, shape=(3,)
                q, orientation quaternion [i, j, k, w], shape=(4,)
            world, World object exposing get_surface_features(),
                get_feature_colors(), get_feature_descriptors(), and
                get_block_bounding_boxes()
            **render_kwargs, additional keyword arguments; any key accepted by
                render() (e.g. background_color, splat_radius, with_distortion,
                feature_output) is forwarded. The special key noise_params (a
                dict of the same tuning knobs as the constructor) overrides the
                camera's noise for this call; noise_params=None disables the
                noise effect.

        Outputs:
            measurement, a dict with keys
                image, (H, W, 3) float32 image in [0, 1]
                keypoints, (M, 2) pixels of visible in-frustum features
                keypoint_depths, (M,) camera-frame z of visible features
                visible_features, (M, 3) world positions of visible features
                visible_mask, (N,) bool over all world features
                projected, (N, 2) pixels of all features (may be out of bounds)
                depth, (N,) camera-frame z of all features
                colors, (N, 3) RGB colors of all features, or None when
                    feature_output is 'descriptors'
                visible_colors, (M, 3) RGB colors of the visible features, or
                    None when feature_output is 'descriptors'
                descriptors, (N, D) generic descriptor vectors (e.g. SIFT/
                    ALIKED) of all features, or None when feature_output is
                    'rgb' or the world has none
                visible_descriptors, (M, D) descriptor vectors of the visible
                    features, or None when feature_output is 'rgb'
        """
        noise_params = render_kwargs.pop('noise_params', _NOISE_UNSET)
        if noise_params is _NOISE_UNSET:
            noise_params = self.noise_params
        else:
            noise_params = _coerce_noise_params(noise_params, default_splat_radius=self.splat_radius)
        measurement = self.render(world, state, **render_kwargs)
        if noise_params is not None:
            self._inject_noise(measurement, noise_params)
        return measurement

    def _inject_noise(self, measurement, noise_params):
        """
        Randomly inject synthetic features onto the rendered image.

        Phantom features are drawn as random-color splats at random in-bounds
        pixels, simulating sensor artifacts / false detections. They carry no
        3D world position, so only measurement['image'] is modified; the
        keypoint/depth/visibility arrays remain clean ground truth.

        Inputs:
            measurement, dict as returned by render()
            noise_params, a coerced dict with keys feature_rate, splat_radius,
                intensity, and an optional seed

        Outputs:
            measurement, the same dict with image modified in place
        """
        rng = np.random.default_rng(noise_params.get('seed'))
        rate = noise_params['feature_rate']
        if rate <= 0:
            return measurement
        K = int(rng.poisson(rate))
        if K == 0:
            return measurement
        radius = noise_params['splat_radius']
        intensity = noise_params['intensity']

        u = rng.uniform(0.0, self.width, size=K)
        v = rng.uniform(0.0, self.height, size=K)
        colors = rng.uniform(0.0, 1.0, size=(K, 3)) * intensity

        image = measurement['image']
        centers_c = np.clip(np.round(u).astype(np.intp), 0, self.width - 1)
        centers_r = np.clip(np.round(v).astype(np.intp), 0, self.height - 1)
        offsets = np.arange(-radius, radius + 1)
        rr = centers_r[:, None] + offsets[None, :]              # (K, Pr)
        cc = centers_c[:, None] + offsets[None, :]              # (K, Pc)
        inside = ((rr >= 0) & (rr < self.height))[:, :, None] & \
                 ((cc >= 0) & (cc < self.width))[:, None, :]
        flat = (np.clip(rr, 0, self.height - 1)[:, :, None] * self.width +
                np.clip(cc, 0, self.width - 1)[:, None, :])[inside]
        values = np.broadcast_to(colors.astype(np.float32)[:, None, None, :],
                                 inside.shape + (3,))[inside]
        image.reshape(-1, 3)[flat] = values
        return measurement

    def world_to_camera(self, points_world, camera_pose):
        """
        Transform world-frame points into the camera frame.

        Inputs:
            points_world, (N, 3) array of world-frame positions
            camera_pose, dict with keys
                x, camera world position, shape=(3,)
                q, world-to-camera quaternion [i, j, k, w], shape=(4,)

        Outputs:
            points_cam, (N, 3) array of camera-frame positions (x right, y down, z forward)
        """
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        R_WC = Rotation.from_quat(np.asarray(camera_pose['q'], dtype=np.float64)).as_matrix()
        return (points_world - camera_pose['x']) @ R_WC.T

    def distort_points(self, points_cam):
        """
        Apply the OpenCV radial/tangential distortion model to normalized
        camera-frame coordinates.

        Inputs:
            points_cam, (N, 3) array of camera-frame points

        Outputs:
            points_distorted, (N, 2) array of distorted normalized coordinates (x', y')
        """
        points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = points_cam[:, 2]
        safe_z = np.where(z != 0, z, 1.0)  # Guard against divide-by-zero.
        x = points_cam[:, 0] / safe_z
        y = points_cam[:, 1] / safe_z

        k1, k2, p1, p2, k3 = self.dist_coeffs
        r2 = x**2 + y**2
        radial = 1.0 + k1*r2 + k2*r2**2 + k3*r2**3
        x_distorted = x*radial + 2*p1*x*y + p2*(r2 + 2*x**2)
        y_distorted = y*radial + p1*(r2 + 2*y**2) + 2*p2*x*y
        return np.stack([x_distorted, y_distorted], axis=-1)

    def project_points(self, points_cam, with_distortion=True):
        """
        Project camera-frame points to pixel coordinates.

        Inputs:
            points_cam, (N, 3) array of camera-frame points
            with_distortion, if True, apply the distortion model

        Outputs:
            pixels, (N, 2) array of pixel coordinates (u, v). Points with
                z <= 0 are left as computed (filtered by the caller).
        """
        points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = points_cam[:, 2]
        safe_z = np.where(z != 0, z, 1.0)  # Guard against divide-by-zero.
        x = points_cam[:, 0] / safe_z
        y = points_cam[:, 1] / safe_z

        if with_distortion:
            x, y = self.distort_points(points_cam).T

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.stack([u, v], axis=-1)

    def compute_occlusion(self, points_world, camera_pose, world):
        """
        Compute which world points are NOT occluded by world blocks.

        A feature is occluded if any block strictly intersects the open segment
        (camera, feature], i.e. the slab ray-tracing parameter t satisfies
        t_entry <= t_exit, t_entry < 1 - 1e-6 and t_exit > 1e-6.

        Inputs:
            points_world, (N, 3) array of world-frame feature positions
            camera_pose, dict with keys
                x, camera world position, shape=(3,)
                q, world-to-camera quaternion [i, j, k, w], shape=(4,)
            world, World object exposing get_block_bounding_boxes() returning a
                list of (xmin, xmax, ymin, ymax, zmin, zmax) extents

        Outputs:
            visible, (N,) boolean array, True where the feature is NOT occluded
        """
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        N = points_world.shape[0]
        visible = np.ones(N, dtype=bool)
        if N == 0:
            return visible

        origin = np.asarray(camera_pose['x'], dtype=np.float64)
        direction = points_world - origin  # (N, 3), t = 1 at the feature

        for extents in world.get_block_bounding_boxes():
            box = np.asarray(extents, dtype=np.float64)
            bmin = box[[0, 2, 4]]
            bmax = box[[1, 3, 5]]

            d_mask = direction != 0.0
            safe_d = np.where(d_mask, direction, 1.0)
            t1 = (bmin - origin) / safe_d  # (3,) / (N, 3) -> (N, 3)
            t2 = (bmax - origin) / safe_d
            # For zero direction components the ray is parallel to the slab:
            # it intersects iff the origin lies inside that slab. If the origin
            # is outside, that axis excludes the ray entirely (t_entry > t_exit).
            in_slab = (origin >= bmin) & (origin <= bmax)  # (3,)
            t1 = np.where(d_mask, t1, np.where(in_slab, -np.inf, np.inf))
            t2 = np.where(d_mask, t2, np.where(in_slab, np.inf, np.inf))

            tmin = np.minimum(t1, t2)
            tmax = np.maximum(t1, t2)
            t_entry = np.max(tmin, axis=1)
            t_exit = np.min(tmax, axis=1)
            hit = (t_entry <= t_exit) & (t_entry < 1 - 1e-6) & (t_exit > 1e-6)
            visible &= ~hit

        return visible

    def render(self, world, state, background_color=None, splat_radius=None, with_distortion=True,
               feature_output=None):
        """
        Render a synthetic image of the world's surface features.

        Inputs:
            world, World object exposing get_surface_features(),
                get_feature_colors(), get_feature_descriptors(), and
                get_block_bounding_boxes()
            state, a dict describing the vehicle state with keys
                x, position, m, shape=(3,)
                q, orientation quaternion [i, j, k, w], shape=(4,)
            background_color, RGB tuple in [0, 1] (default [0.9, 0.9, 0.9])
            splat_radius, features are splatted on a (2*splat_radius+1) square
                patch centered on their rounded pixel; None uses the camera's
                instance default (see __init__)
            with_distortion, if True, apply the distortion model
            feature_output, per-call override of the camera's feature_output
                setting ('all', 'rgb', or 'descriptors'); None uses the instance
                default. See __init__ for the semantics.

        Outputs:
            render, a dict with keys
                image, (H, W, 3) float32 image in [0, 1]
                keypoints, (M, 2) pixels of visible in-frustum features
                keypoint_depths, (M,) camera-frame z of visible features
                visible_features, (M, 3) world positions of visible features
                visible_mask, (N,) bool over all world features
                projected, (N, 2) pixels of all features (may be out of bounds)
                depth, (N,) camera-frame z of all features
                colors, (N, 3) RGB colors of all features, or None if
                    feature_output is 'descriptors' (or the world has no colors)
                visible_colors, (M, 3) RGB colors of the visible features, or None
                descriptors, (N, D) generic descriptor vectors (e.g. SIFT/
                    ALIKED) of all features, or None if feature_output is 'rgb'
                    or the world has none
                visible_descriptors, (M, D) descriptor vectors of the visible
                    features, or None
        """
        if splat_radius is None:
            splat_radius = self.splat_radius
        if background_color is None:
            background_color = [0.9, 0.9, 0.9]
        feature_output = _resolve_feature_output(feature_output, self.feature_output)
        include_colors = feature_output in ('all', 'rgb')
        include_descriptors = feature_output in ('all', 'descriptors')
        # A scalar fill is dramatically faster than broadcasting an RGB list
        # into the array (np.full dispatches the latter through a slow
        # strided-copy path), so take that route whenever channels match.
        bg = np.asarray(background_color, dtype=np.float32).ravel()
        if bg.size == 3 and np.all(bg == bg[0]):
            image = np.full((self.height, self.width, 3), float(bg[0]), dtype=np.float32)
        else:
            image = np.empty((self.height, self.width, 3), dtype=np.float32)
            image[:] = background_color

        features = None
        getter = getattr(world, 'get_surface_features', None)
        if getter is not None:
            features = getter()
        if features is None:
            features = np.empty((0, 3), dtype=np.float64)

        colors = None
        getter = getattr(world, 'get_feature_colors', None)
        if getter is None:
            getter = getattr(world, 'get_feature_descriptors', None)  # legacy worlds: RGB colors
        if getter is not None:
            colors = getter()

        # Generic descriptor vectors (e.g. SIFT/ALIKED) carried alongside the
        # RGB colors; these are passthrough outputs, not used in rendering.
        descriptors = None
        getter = getattr(world, 'get_feature_descriptors', None)
        if getter is not None:
            descriptors = getter()

        features = np.asarray(features, dtype=np.float64).reshape(-1, 3)
        N = features.shape[0]

        camera_pose = self.compute_camera_pose(state)

        if N == 0:
            empty_kp = np.empty((0, 2), dtype=np.float64)
            return {
                'image': image,
                'keypoints': empty_kp,
                'keypoint_depths': np.empty(0, dtype=np.float64),
                'visible_features': np.empty((0, 3), dtype=np.float64),
                'visible_mask': np.zeros(0, dtype=bool),
                'projected': np.empty((0, 2), dtype=np.float64),
                'depth': np.empty(0, dtype=np.float64),
                'colors': None,
                'visible_colors': None,
                'descriptors': None,
                'visible_descriptors': None,
            }

        points_cam = self.world_to_camera(features, camera_pose)
        depth = points_cam[:, 2]
        projected = self.project_points(points_cam, with_distortion=with_distortion)
        u = projected[:, 0]
        v = projected[:, 1]
        not_occluded = self.compute_occlusion(features, camera_pose, world)

        visible_mask = (depth > self.near_plane) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height) & not_occluded

        if colors is None:
            splat_colors = np.full((N, 3), 0.6, dtype=np.float64)
        else:
            splat_colors = np.asarray(colors, dtype=np.float64)
            if splat_colors.shape != (N, 3):
                splat_colors = np.full((N, 3), 0.6, dtype=np.float64)
            else:
                splat_colors = np.clip(splat_colors, 0.0, 1.0)

        colors_out = splat_colors if include_colors else None

        if include_descriptors and descriptors is not None:
            desc_out = np.asarray(descriptors, dtype=np.float64)
            if desc_out.ndim != 2 or desc_out.shape[0] != N:
                desc_out = None
        else:
            desc_out = None

        visible_idx = np.nonzero(visible_mask)[0]
        if visible_idx.size > 0:
            # Splat every visible feature onto the image in one vectorized
            # write. Pixels are flattened to (K, Pr, Pc) indices in feature
            # order so that later features overwrite earlier ones where
            # splats overlap -- same semantics as the per-feature loop.
            centers_r = np.clip(np.round(v[visible_idx]).astype(np.intp), 0, self.height - 1)
            centers_c = np.clip(np.round(u[visible_idx]).astype(np.intp), 0, self.width - 1)
            offsets = np.arange(-int(splat_radius), int(splat_radius) + 1)
            rr = centers_r[:, None] + offsets[None, :]              # (K, Pr)
            cc = centers_c[:, None] + offsets[None, :]              # (K, Pc)
            inside = ((rr >= 0) & (rr < self.height))[:, :, None] & \
                     ((cc >= 0) & (cc < self.width))[:, None, :]
            flat = (np.clip(rr, 0, self.height - 1)[:, :, None] * self.width +
                    np.clip(cc, 0, self.width - 1)[:, None, :])[inside]
            values = np.broadcast_to(splat_colors[visible_idx].astype(np.float32)[:, None, None, :],
                                     inside.shape + (3,))[inside]
            image.reshape(-1, 3)[flat] = values

        return {
            'image': image,
            'keypoints': projected[visible_idx],
            'keypoint_depths': depth[visible_idx],
            'visible_features': features[visible_idx],
            'visible_mask': visible_mask,
            'projected': projected,
            'depth': depth,
            'colors': colors_out,
            'visible_colors': (colors_out[visible_idx] if colors_out is not None
                               else None),
            'descriptors': desc_out,
            'visible_descriptors': (desc_out[visible_idx] if desc_out is not None
                                    else None),
        }


def _per_drone_list(value, num_drones, name):
    """
    Normalize a parameter into a list of exactly num_drones entries.

    A dict or scalar is broadcast to every drone; a sequence (list, tuple, or
    array) must already have length num_drones, which is the maximum batch size
    of the batched camera.

    Inputs:
        value, a shared dict/scalar or a per-drone sequence
        num_drones, the maximum batch size
        name, parameter name for error messages

    Outputs:
        params, list of length num_drones
    """
    if isinstance(value, dict) or isinstance(value, (int, float, np.integer, np.floating)):
        return [value for _ in range(num_drones)]
    if isinstance(value, (list, tuple, np.ndarray)):
        seq = list(value)
        if len(seq) != num_drones:
            raise ValueError("{} must be shared (scalar/dict) or have length num_drones={}, got {}".format(
                name, num_drones, len(seq)))
        return seq
    raise ValueError("{} must be a scalar/dict (shared) or a sequence of {} entries, got {}".format(
        name, num_drones, type(value).__name__))


def _coerce_noise_params_list(noise_params, num_drones, default_splat_radius):
    """
    Normalize noise_params into a per-drone list of coerced dicts (or None).

    Inputs:
        noise_params, None, a single dict (broadcast), or a sequence of
            num_drones dicts or None entries (None disables the effect for that
            drone)
        num_drones, the maximum batch size
        default_splat_radius, a shared scalar or per-drone sequence used when an
            entry omits 'splat_radius'

    Outputs:
        params, list of length num_drones of coerced dicts (see
            _coerce_noise_params) or None
    """
    if noise_params is None:
        return [None] * num_drones

    if isinstance(noise_params, dict):
        seq = [noise_params] * num_drones
    elif isinstance(noise_params, (list, tuple, np.ndarray)):
        seq = list(noise_params)
        if len(seq) != num_drones:
            raise ValueError("noise_params sequence must have length num_drones={}, got {}".format(
                num_drones, len(seq)))
    else:
        raise ValueError("noise_params must be a dict, a sequence of dicts/None, or None, got {}".format(
            type(noise_params).__name__))

    default_radii = _per_drone_list(default_splat_radius, num_drones, 'splat_radius')
    out = []
    for b, entry in enumerate(seq):
        if entry is None:
            out.append(None)
        else:
            out.append(_coerce_noise_params(entry, default_splat_radius=float(default_radii[b])))
    return out


def randomize_camera_params(num_drones=1, intrinsics=None, extrinsics=None,
                            intrinsics_scale=0.1, principal_point_scale=0.05,
                            dist_coeffs_scale=0.0, extrinsics_translation=0.1,
                            extrinsics_rotation=0.2, noise_fraction=0.0,
                            feature_rate_range=(5.0, 40.0), seed=None):
    """
    Sample a batch of per-drone camera parameters for domain randomization.

    Draws num_drones perturbed intrinsics/extrinsics around the given bases and
    optionally turns on the measurement() visual noise effect for a fraction of
    the drones. Intended to feed the per-drone sequences accepted by
    BatchedPinholeCamera; the image resolution is preserved so the whole batch
    is valid together.

    Inputs:
        num_drones, number of per-drone parameter sets to sample
        intrinsics, base intrinsics dict (defaults to a 640x480 pinhole camera)
        extrinsics, base extrinsics dict (defaults to identity body-to-camera)
        intrinsics_scale, relative standard deviation of the log-normal scale
            applied to the focal lengths
        principal_point_scale, principal point jitter as a fraction of the image
            width/height (standard deviation of the additive offset)
        dist_coeffs_scale, standard deviation of the zero-mean jitter added to
            each distortion coefficient
        extrinsics_translation, standard deviation (m) of the additive position
            jitter, per axis
        extrinsics_rotation, standard deviation (rad) of the random small
            rotation applied to the extrinsics orientation
        noise_fraction, fraction of drones that receive a synthesized
            measurement-noise params dict (the rest get None)
        feature_rate_range, (lo, hi) uniform range for the sampled feature_rate
            of noisy drones
        seed, optional int making the whole sampling reproducible

    Outputs:
        intrinsics_list, extrinsics_list, noise_params_list, three lists of
            length num_drones of intrinsics dicts, extrinsics dicts, and
            noise params dicts (or None), suitable for the corresponding
            BatchedPinholeCamera constructor arguments
    """
    if intrinsics is None:
        intrinsics = {'fx': 500.0, 'fy': 500.0, 'width': 640, 'height': 480, 'cx': 320.0, 'cy': 240.0,
                      'dist_coeffs': np.array([0.0, 0.0, 0.0, 0.0, 0.0])}  # [k1, k2, p1, p2, k3]
    if extrinsics is None:
        extrinsics = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.0, 0.0, 0.0, 1.0])}

    for name, value in (('intrinsics_scale', intrinsics_scale),
                        ('principal_point_scale', principal_point_scale),
                        ('dist_coeffs_scale', dist_coeffs_scale),
                        ('extrinsics_translation', extrinsics_translation),
                        ('extrinsics_rotation', extrinsics_rotation)):
        if value < 0.0:
            raise ValueError("{} must be >= 0, got {}".format(name, value))
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be in [0, 1], got {}".format(noise_fraction))
    feature_rate_range = tuple(feature_rate_range)
    if len(feature_rate_range) != 2 or feature_rate_range[1] <= feature_rate_range[0]:
        raise ValueError("feature_rate_range must be a (lo, hi) pair with hi > lo, got {}".format(feature_rate_range))

    rng = np.random.default_rng(seed)
    width = int(intrinsics['width'])
    height = int(intrinsics['height'])
    base_fx = float(intrinsics['fx'])
    base_fy = float(intrinsics['fy'])
    base_cx = float(intrinsics['cx'])
    base_cy = float(intrinsics['cy'])
    base_dist = np.asarray(intrinsics['dist_coeffs'], dtype=np.float64)
    base_pos = np.asarray(extrinsics['position'], dtype=np.float64)
    base_R = Rotation.from_quat(np.asarray(extrinsics['orientation'], dtype=np.float64))

    intrinsics_list, extrinsics_list, noise_params_list = [], [], []
    for _ in range(num_drones):
        fx = base_fx * (1.0 + rng.normal(0.0, intrinsics_scale))
        fy = base_fy * (1.0 + rng.normal(0.0, intrinsics_scale))
        cx = base_cx + width * rng.normal(0.0, principal_point_scale)
        cy = base_cy + height * rng.normal(0.0, principal_point_scale)
        dist = base_dist + rng.normal(0.0, dist_coeffs_scale, size=5)
        intrinsics_list.append({'fx': fx, 'fy': fy, 'width': width, 'height': height,
                                'cx': cx, 'cy': cy, 'dist_coeffs': dist})

        position = base_pos + rng.normal(0.0, extrinsics_translation, size=3)
        axis = rng.normal(size=3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm
        theta = abs(float(rng.normal(0.0, extrinsics_rotation)))
        orientation = (Rotation.from_rotvec(axis * theta) * base_R).as_quat()
        extrinsics_list.append({'position': position, 'orientation': orientation})

        if rng.random() < noise_fraction:
            noise_params_list.append({'feature_rate': float(rng.uniform(feature_rate_range[0], feature_rate_range[1])),
                                      'splat_radius': int(rng.integers(1, 4)),
                                      'intensity': float(rng.uniform(0.5, 1.0)),
                                      'seed': int(rng.integers(0, 2**31 - 1))})
        else:
            noise_params_list.append(None)

    return intrinsics_list, extrinsics_list, noise_params_list


class BatchedPinholeCamera:
    """
    Torch-parallelized version of PinholeCamera that renders a batch of
    drones simultaneously.

    Camera parameters can be given either as a single shared value (broadcast
    to every drone) or per drone, which is how domain randomization is applied:
    each drone in a batch can have its own intrinsics (focal lengths, principal
    point, distortion), extrinsics (mount position/orientation), near_plane,
    splat_radius, and noise_params. The image resolution (width/height) must be
    shared across the batch so the rendered images stay a single (B, H, W, 3)
    tensor.

    Each drone may also observe its own environment: pass a single World to
    render()/measurement() for a shared world (the all-feature outputs stay
    stacked (B, N, ...) tensors), or a list/tuple of length B of Worlds (None
    gives a drone an empty world) for per-drone environments. Because the
    per-drone feature counts N can differ, with per-drone worlds the all-feature
    outputs (visible_mask, depth, colors, descriptors) are returned as lists of
    length B; keypoints/visible_* were already per-drone lists.

    Parameters:
        num_drones, number of drones in the batch (the actual batch size B
            passed to render() may be <= num_drones)
        intrinsics, dict of camera intrinsics (shared) or a sequence of
            num_drones dicts (per drone); all entries must share width/height.
        extrinsics, dict of camera extrinsics (shared) or a sequence of
            num_drones dicts (per drone), see PinholeCamera
        near_plane, scalar (shared) or per-drone sequence of minimum
            camera-frame z for a feature to be rendered, m
        device, torch device string, e.g. 'cpu' or 'cuda'
        noise_params, None, a single dict (shared), or a per-drone sequence of
            dict/None (None disables the effect for that drone), enabling the
            measurement() visual noise effect, see PinholeCamera.__init__ for
            the tuning knobs.
        feature_output, which per-feature data measurement()/render() return:
            'all' (default) for both RGB colors and descriptor vectors, 'rgb'
            for colors only, or 'descriptors' for descriptor vectors only. See
            PinholeCamera.__init__ for details.
        splat_radius, scalar (shared) or per-drone sequence of default splat
            radii, pixels; can be overridden per call.
    """
    def __init__(self, num_drones, intrinsics=None, extrinsics=None, near_plane=0.05, device='cpu',
                 noise_params=None, feature_output='all', splat_radius=1, frame_rate=None):
        """
        Parameters:
            num_drones, number of drones in the batch
            intrinsics, dict of camera intrinsics (shared) or a sequence of
                num_drones dicts (per drone); all entries must agree on
                width/height.
            extrinsics, dict of camera extrinsics (shared) or a sequence of
                num_drones dicts (per drone), see PinholeCamera
            near_plane, scalar (shared) or per-drone sequence of minimum
                camera-frame z for a feature to be rendered, m
            device, torch device string
            noise_params, None, a single dict (shared), or a per-drone sequence
                of dict/None (None turns the effect off for that drone): the
                measurement() visual noise effect tuning knobs, which randomly
                injects synthetic features onto each frame independently per
                drone. See PinholeCamera.__init__ for the knob descriptions.
            feature_output, which per-feature data render()/measurement() put
                in their output dicts ('all', 'rgb', or 'descriptors'),
                suppressing colors or descriptors to save memory. Same
                semantics and per-call override as PinholeCamera.__init__.
            splat_radius, scalar (shared) or per-drone sequence of default
                splat radii, pixels; can be overridden per render() call.
            frame_rate, the rate at which the camera captures frames, Hz. Used
                by simulate_batch() to decimate captures from the simulation
                rate. None (default) captures a frame every simulation step.
        """
        if torch is None:
            raise ImportError("torch required for BatchedPinholeCamera")
        if num_drones < 1:
            raise ValueError("num_drones must be >= 1, got {}".format(num_drones))

        self.num_drones = num_drones
        self.device = torch.device(device)
        self.feature_output = _resolve_feature_output(feature_output, 'all')
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        if intrinsics is None:
            intrinsics = {'fx': 500.0, 'fy': 500.0, 'width': 640, 'height': 480, 'cx': 320.0, 'cy': 240.0,
                          'dist_coeffs': np.array([0.0, 0.0, 0.0, 0.0, 0.0])}  # [k1, k2, p1, p2, k3]
        if extrinsics is None:
            extrinsics = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.0, 0.0, 0.0, 1.0])}

        # Per-drone intrinsics. Resolution must be shared so the batch images
        # stay a single (B, H, W, 3) tensor; focal lengths, principal point,
        # and distortion may vary per drone for domain randomization.
        fxs, fys, cxs, cys, widths, heights, dists = [], [], [], [], [], [], []
        for entry in _per_drone_list(intrinsics, num_drones, 'intrinsics'):
            fxs.append(float(entry['fx']))
            fys.append(float(entry['fy']))
            cxs.append(float(entry['cx']))
            cys.append(float(entry['cy']))
            widths.append(int(entry['width']))
            heights.append(int(entry['height']))
            dc = np.asarray(entry['dist_coeffs'], dtype=np.float64)
            if dc.shape != (5,):
                raise ValueError("intrinsics['dist_coeffs'] must have shape (5,), got {}".format(dc.shape))
            dists.append(dc)
        if len(set(widths)) > 1 or len(set(heights)) > 1:
            raise ValueError("image width/height must be shared across the batch so the images stay a single "
                             "(B, H, W, 3) tensor; got per-drone resolutions")
        self.width = widths[0]
        self.height = heights[0]
        self.fx = torch.tensor(fxs, dtype=torch.float32, device=self.device)
        self.fy = torch.tensor(fys, dtype=torch.float32, device=self.device)
        self.cx = torch.tensor(cxs, dtype=torch.float32, device=self.device)
        self.cy = torch.tensor(cys, dtype=torch.float32, device=self.device)
        self.dist_coeffs = torch.tensor(np.stack(dists, axis=0), dtype=torch.float32, device=self.device)  # (num_drones, 5)

        # Per-drone extrinsics: body-to-camera position and rotation.
        positions, rotations = [], []
        for entry in _per_drone_list(extrinsics, num_drones, 'extrinsics'):
            p_BC = np.asarray(entry['position'], dtype=np.float64)
            q_BC = np.asarray(entry['orientation'], dtype=np.float64)
            if p_BC.shape != (3,):
                raise ValueError("extrinsics['position'] must have shape (3,), got {}".format(p_BC.shape))
            if q_BC.shape != (4,):
                raise ValueError("extrinsics['orientation'] must have shape (4,), got {}".format(q_BC.shape))
            positions.append(p_BC)
            rotations.append(Rotation.from_quat(q_BC).as_matrix())
        self.p_BC = torch.tensor(np.stack(positions, axis=0), dtype=torch.float32, device=self.device)  # (num_drones, 3)
        self.R_BC = torch.tensor(np.stack(rotations, axis=0), dtype=torch.float32,
                                 device=self.device)  # (num_drones, 3, 3) body -> camera

        # Per-drone near plane.
        if near_plane is None:
            near_plane = 0.05
        self.near_plane = torch.tensor([float(v) for v in _per_drone_list(near_plane, num_drones, 'near_plane')],
                                       dtype=torch.float32, device=self.device)  # (num_drones,)

        # Per-drone splat radius.
        self.splat_radius = [int(r) for r in _per_drone_list(splat_radius, num_drones, 'splat_radius')]
        if any(r < 0 for r in self.splat_radius):
            raise ValueError("splat_radius must be non-negative, got {}".format(self.splat_radius))

        self.noise_params = _coerce_noise_params_list(noise_params, num_drones, self.splat_radius)
        self.frame_rate = frame_rate

    @staticmethod
    def _quat_to_rotmat(q):
        """
        Convert a batch of [i, j, k, w] quaternions to rotation matrices.

        Inputs:
            q, (B, 4) tensor of quaternions [i, j, k, w]

        Outputs:
            R, (B, 3, 3) tensor of rotation matrices
        """
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        B = q.shape[0]
        R = torch.empty((B, 3, 3), device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        return R

    def measurement(self, worlds, states, **render_kwargs):
        """
        Compute camera measurements for a batch of drones.

        This is the primary sensor interface, consistent with the measurement()
        pattern used by other RotorPy sensors (IMU, mocap, range sensors). The
        raw render() output is produced first, then optional measurement
        effects are layered on top. The visual noise effect (see noise_params
        in __init__) randomly injects synthetic features onto each frame when
        enabled for that drone; additional effects can be added here later.

        Inputs:
            worlds, a single World object shared by all drones, or a
                list/tuple of length B of Worlds (None for an empty world) for
                per-drone environments, see render() for the output shapes.
            states, a dict describing the vehicle states with keys
                x, position, (B, 3) tensor
                q, orientation quaternion [i, j, k, w], (B, 4) tensor
            **render_kwargs, additional keyword arguments; any key accepted by
                render() (e.g. background_color, splat_radius, with_distortion,
                feature_output) is forwarded. The special key noise_params
                (None, a single dict, or a per-drone sequence of dict/None with
                the same tuning knobs as the constructor) overrides the
                camera's noise for this call; None disables the noise effect.

        Outputs:
            measurement, a dict with the same keys and shapes as render().
        """
        noise_params = render_kwargs.pop('noise_params', _NOISE_UNSET)
        if noise_params is _NOISE_UNSET:
            noise_params = self.noise_params
        else:
            noise_params = _coerce_noise_params_list(noise_params, self.num_drones, self.splat_radius)
        measurement = self.render(worlds, states, **render_kwargs)
        if any(params is not None for params in noise_params):
            self._inject_noise_batched(measurement, noise_params)
        return measurement

    def _inject_noise_batched(self, measurement, noise_params_list):
        """
        Randomly inject synthetic features onto each drone's rendered frame.

        Identical semantics to PinholeCamera._inject_noise, but applied
        independently per drone using that drone's own noise_params (None, or
        feature_rate 0, turns the effect off for that drone). When an entry
        carries a seed, the per-drone stream is seeded with (seed, drone index)
        so the whole batch is reproducible while each drone still gets its own
        sequence. Only the image tensor is modified; keypoint/depth/visibility
        outputs stay clean ground truth.

        Inputs:
            measurement, dict as returned by BatchedPinholeCamera.render()
            noise_params_list, list of length num_drones of coerced dicts (with
                keys feature_rate, splat_radius, intensity, optional seed) or
                None

        Outputs:
            measurement, the same dict with image modified in place
        """
        B, H, W, _ = measurement['image'].shape
        b_list, r_list, c_list, color_list, radius_list = [], [], [], [], []
        for b in range(B):
            params = noise_params_list[b]
            if params is None or params['feature_rate'] <= 0:
                continue
            seed = params.get('seed')
            rng = np.random.default_rng((seed, b) if seed is not None else None)
            K = int(rng.poisson(params['feature_rate']))
            if K == 0:
                continue
            u = rng.uniform(0.0, W, size=K)
            v = rng.uniform(0.0, H, size=K)
            colors = rng.uniform(0.0, 1.0, size=(K, 3)) * params['intensity']
            b_list.append(np.full(K, b, dtype=np.int64))
            r_list.append(np.round(v).astype(np.int64))
            c_list.append(np.round(u).astype(np.int64))
            color_list.append(colors)
            radius_list.append(np.full(K, params['splat_radius'], dtype=np.int64))
        if not b_list:
            return measurement

        b_idx = torch.from_numpy(np.concatenate(b_list)).to(self.device)
        r = torch.from_numpy(np.concatenate(r_list)).to(self.device)
        c = torch.from_numpy(np.concatenate(c_list)).to(self.device)
        radii = torch.from_numpy(np.concatenate(radius_list)).to(self.device)
        color_src = torch.from_numpy(np.concatenate(color_list, axis=0).astype(np.float32)).to(self.device)
        self._splat_batched(measurement['image'], b_idx, r, c, radii, color_src)
        return measurement

    @staticmethod
    def _splat_batched(image, b_idx, r, c, radii, colors):
        """
        Splat square color patches onto a (B, H, W, 3) image tensor.

        Each feature is drawn on a (2*radius+1)^2 patch centered on its rounded
        pixel. Radii may differ between features; where patches overlap the
        winner is resolved deterministically: splats with a larger radius are
        drawn on top, and within equal radii later features overwrite earlier
        ones. (The group-by-group writes are de-duplicated explicitly because
        PyTorch's index_put_ on overlapping indices is not deterministic.)

        Inputs:
            image, (B, H, W, 3) float tensor, modified in place
            b_idx, (K,) drone index per feature
            r, (K,) pixel row centers (unclamped)
            c, (K,) pixel column centers (unclamped)
            radii, (K,) integer splat radius per feature
            colors, (K, 3) RGB colors
        """
        B, H, W, _ = image.shape
        if b_idx.numel() == 0:
            return
        flat_all, value_all = [], []
        for radius in sorted({int(x) for x in torch.unique(radii).tolist()}):
            group = radii == radius
            bi, ri, ci, co = b_idx[group], r[group], c[group], colors[group]
            dr = torch.arange(-radius, radius + 1, device=image.device)
            dc = torch.arange(-radius, radius + 1, device=image.device)
            rows = ri[:, None, None] + dr[None, :, None]      # (K, Pr, 1)
            cols = ci[:, None, None] + dc[None, None, :]      # (K, 1, Pc)
            inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
            rows = rows.clamp(0, H - 1)
            cols = cols.clamp(0, W - 1)
            full_idx = bi[:, None, None] * (H * W) + rows * W + cols  # (K, Pr, Pc)
            full_idx = full_idx[inside]
            values = co[:, None, None, :].expand(-1, rows.shape[1], cols.shape[2], -1)[inside]
            flat_all.append(full_idx)
            value_all.append(values)
        if not flat_all:
            return
        flat = torch.cat(flat_all)
        values = torch.cat(value_all)
        # Stable-sort by flat pixel index: within an equal-index run, pixels
        # stay in (radius ascending, feature order) so the last element of each
        # run is the deterministic winner (largest radius, latest feature).
        order = torch.argsort(flat, stable=True)
        flat_s = flat[order]
        val_s = values[order]
        same = flat_s[1:] == flat_s[:-1]
        last_of_run = torch.cat([~same, torch.tensor([True], device=flat.device)])
        flat = image.reshape(B * H * W, 3)
        flat.index_put_((flat_s[last_of_run],), val_s[last_of_run])

    @staticmethod
    def _compute_not_occluded(origin, directions, extents_list, device):
        """
        Determine which ray segments are NOT occluded by the given blocks.

        A point is occluded if any block strictly intersects the open segment
        (origin, origin+direction], using the same slab ray-tracing rule as
        PinholeCamera.compute_occlusion. Vectorized over the trailing feature
        dimension; origin and directions must share all preceding dimensions.

        Inputs:
            origin, (..., 3) ray origins
            directions, (..., M, 3) vectors from each origin to each feature
            extents_list, iterable of (xmin, xmax, ymin, ymax, zmin, zmax)
            device, torch device

        Outputs:
            not_occluded, (..., M) bool tensor, True where the feature is
                visible (not occluded)
        """
        not_occluded = torch.ones(directions.shape[:-1], dtype=torch.bool, device=device)
        o = origin.unsqueeze(-2)  # (..., 1, 3)
        for extents in extents_list:
            box = np.asarray(extents, dtype=np.float64)
            bmin = torch.tensor(box[[0, 2, 4]], dtype=torch.float32, device=device)
            bmax = torch.tensor(box[[1, 3, 5]], dtype=torch.float32, device=device)
            d_mask = directions != 0.0
            safe_d = torch.where(d_mask, directions, torch.ones_like(directions))
            t1 = (bmin - o) / safe_d
            t2 = (bmax - o) / safe_d
            in_slab = (o >= bmin) & (o <= bmax)
            inf = torch.tensor(float('inf'), dtype=torch.float32, device=device)
            t1 = torch.where(d_mask, t1, torch.where(in_slab, -inf, inf))
            t2 = torch.where(d_mask, t2, torch.where(in_slab, inf, inf))
            tmin = torch.minimum(t1, t2)
            tmax = torch.maximum(t1, t2)
            t_entry = torch.max(tmin, dim=-1).values
            t_exit = torch.min(tmax, dim=-1).values
            hit = (t_entry <= t_exit) & (t_entry < 1 - 1e-6) & (t_exit > 1e-6)
            not_occluded &= ~hit
        return not_occluded

    def render(self, worlds, states, background_color=None, splat_radius=None, with_distortion=True,
               feature_output=None):
        """
        Render synthetic images for a batch of drones.

        Inputs:
            worlds, a single World object shared by all drones in the batch
                (the all-feature outputs come back as stacked (B, N, ...)
                tensors), or a list/tuple of length B of World objects (None
                gives a drone an empty world) for per-drone environments, in
                which case the per-feature outputs (visible_mask, depth, colors,
                descriptors) are lists of length B because the per-drone feature
                count N can differ. A world must expose get_surface_features(),
                get_feature_colors(), get_feature_descriptors(), and
                get_block_bounding_boxes().
            states, a dict describing the vehicle states with keys
                x, position, (B, 3) tensor
                q, orientation quaternion [i, j, k, w], (B, 4) tensor
            background_color, RGB tuple in [0, 1] (default [0.9, 0.9, 0.9])
            splat_radius, features are splatted on a (2*splat_radius+1) square
                patch centered on their rounded pixel; a scalar (shared) or a
                per-drone sequence, or None to use the camera's per-drone
                defaults
            with_distortion, if True, apply each drone's distortion model
            feature_output, per-call override of the camera's feature_output
                setting ('all', 'rgb', or 'descriptors'); None uses the instance
                default. See PinholeCamera.__init__ for the semantics.

        Outputs:
            render, a dict with keys
                image, (B, H, W, 3) float tensor in [0, 1]
                visible_mask, (B, N) bool tensor (shared world) or list of
                    length B of (N_b,) bool tensors (per-drone worlds) over all
                    features
                depth, (B, N) camera-frame z tensor (shared world) or list of
                    length B of (N_b,) tensors (per-drone worlds)
                projected, (B, N, 2) pixels of all features (may be out of
                    bounds), or list of length B of (N_b, 2) tensors (per-drone
                    worlds)
                keypoints, list of length B of (M_b, 2) pixel tensors
                keypoint_depths, list of length B of (M_b,) depth tensors
                visible_features, list of length B of (M_b, 3) world-position tensors
                colors, (B, N, 3) tensor or list of length B of (N_b, 3) RGB
                    color tensors, or None when feature_output is 'descriptors'
                visible_colors, list of length B of (M_b, 3) RGB color tensors,
                    or None when feature_output is 'descriptors'
                descriptors, (B, N, D) tensor or list of length B of (N_b, D)
                    generic descriptor vectors (e.g. SIFT/ALIKED), or None when
                    feature_output is 'rgb' or a world has none
                visible_descriptors, list of length B of (M_b, D) descriptor
                    tensors, or None when a world has none
        """
        if background_color is None:
            background_color = [0.9, 0.9, 0.9]
        feature_output = _resolve_feature_output(feature_output, self.feature_output)
        include_colors = feature_output in ('all', 'rgb')
        include_descriptors = feature_output in ('all', 'descriptors')

        x = torch.as_tensor(states['x']).float().to(self.device)
        q = torch.as_tensor(states['q']).float().to(self.device)
        B = x.shape[0]
        if B > self.num_drones:
            raise ValueError("batch size B={} exceeds num_drones={}".format(B, self.num_drones))
        if x.shape != (B, 3):
            raise ValueError("states['x'] must have shape (B, 3), got {}".format(tuple(x.shape)))
        if q.shape != (B, 4):
            raise ValueError("states['q'] must have shape (B, 4), got {}".format(tuple(q.shape)))

        # Per-drone camera parameters for this batch.
        fx, fy, cx, cy = self.fx[:B], self.fy[:B], self.cx[:B], self.cy[:B]
        dc = self.dist_coeffs[:B]  # (B, 5)
        p_BC, R_BC = self.p_BC[:B], self.R_BC[:B]
        near = self.near_plane[:B]  # (B,)

        if splat_radius is None:
            sp = [self.splat_radius[b] for b in range(B)]
        else:
            sp = [int(v) for v in _per_drone_list(splat_radius, B, 'splat_radius')]
        sp_t = torch.tensor(sp, dtype=torch.long, device=self.device)  # (B,) per-drone radii

        # World-to-camera rotation per drone: R_WC = R_BC @ R_WB^T.
        R_WB = self._quat_to_rotmat(q)  # (B, 3, 3), body -> world
        R_WC = torch.einsum('bij,bjk->bik', R_BC, R_WB.transpose(-1, -2))  # (B, 3, 3)
        # Camera world position per drone: p_WC = x + R_WB @ p_BC.
        p_WC = x + torch.einsum('bij,bj->bi', R_WB, p_BC)  # (B, 3)

        image = torch.full((B, self.height, self.width, 3), float(background_color[0]),
                           dtype=torch.float32, device=self.device)
        image[:, :, :, 0] = background_color[0]
        image[:, :, :, 1] = background_color[1]
        image[:, :, :, 2] = background_color[2]

        if isinstance(worlds, (list, tuple)):
            return self._render_per_drone_worlds(worlds, B, image, fx, fy, cx, cy, dc, near, sp_t,
                                                 R_WC, p_WC, include_colors, include_descriptors,
                                                 with_distortion)
        return self._render_shared_world(worlds, B, image, fx, fy, cx, cy, dc, near, sp_t,
                                         R_WC, p_WC, include_colors, include_descriptors,
                                         with_distortion)

    def _render_shared_world(self, world, B, image, fx, fy, cx, cy, dc, near, sp_t,
                             R_WC, p_WC, include_colors, include_descriptors, with_distortion):
        """
        The single-world render path: all drones observe the same environment,
        so features are shared and the fixed-size outputs stay (B, N, ...).
        """
        features = None
        getter = getattr(world, 'get_surface_features', None)
        if getter is not None:
            features = getter()
        if features is None:
            features = np.empty((0, 3), dtype=np.float64)

        colors = None
        getter = getattr(world, 'get_feature_colors', None)
        if getter is None:
            getter = getattr(world, 'get_feature_descriptors', None)  # legacy worlds: RGB colors
        if getter is not None:
            colors = getter()

        # Generic descriptor vectors (e.g. SIFT/ALIKED) carried alongside the
        # RGB colors; these are passthrough outputs, not used in rendering.
        descriptors = None
        getter = getattr(world, 'get_feature_descriptors', None)
        if getter is not None:
            descriptors = getter()

        features_np = np.asarray(features, dtype=np.float64).reshape(-1, 3)
        N = features_np.shape[0]
        features_t = torch.tensor(features_np, dtype=torch.float32, device=self.device)

        empty_kp = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        empty_d = torch.empty((0,), dtype=torch.float32, device=self.device)
        empty_f = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        if N == 0:
            return {
                'image': image,
                'visible_mask': torch.zeros((B, 0), dtype=torch.bool, device=self.device),
                'depth': torch.zeros((B, 0), dtype=torch.float32, device=self.device),
                'projected': torch.empty((B, 0, 2), dtype=torch.float32, device=self.device),
                'keypoints': [empty_kp for _ in range(B)],
                'keypoint_depths': [empty_d for _ in range(B)],
                'visible_features': [empty_f for _ in range(B)],
                'colors': None,
                'visible_colors': None,
                'descriptors': None,
                'visible_descriptors': None,
            }

        # Camera-frame points: p_c = R_WC @ (p_W - p_WC).
        points_cam = torch.einsum('bij,bnj->bni', R_WC, features_t[None] - p_WC[:, None])  # (B, N, 3)
        depth = points_cam[:, :, 2]

        # Projection with per-drone intrinsics/distortion.
        z = depth
        safe_z = torch.where(z != 0, z, torch.ones_like(z))
        x_n = points_cam[:, :, 0] / safe_z
        y_n = points_cam[:, :, 1] / safe_z
        if with_distortion:
            k1, k2, p1, p2, k3 = dc[:, 0:1], dc[:, 1:2], dc[:, 2:3], dc[:, 3:4], dc[:, 4:5]
            r2 = x_n**2 + y_n**2
            radial = 1.0 + k1*r2 + k2*r2**2 + k3*r2**3
            xd = x_n*radial + 2*p1*x_n*y_n + p2*(r2 + 2*x_n**2)
            yd = y_n*radial + p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n
        else:
            xd, yd = x_n, y_n
        u = fx[:, None] * xd + cx[:, None]
        v = fy[:, None] * yd + cy[:, None]

        # Occlusion (loop over blocks, vectorized over (B, N)).
        not_occluded = self._compute_not_occluded(p_WC, features_t[None] - p_WC[:, None],
                                                  world.get_block_bounding_boxes(), self.device)

        visible_mask = (depth > near[:, None]) & (u >= 0) & (u < self.width) & \
                       (v >= 0) & (v < self.height) & not_occluded

        if colors is None:
            splat_colors = torch.full((N, 3), 0.6, dtype=torch.float32, device=self.device)
        else:
            colors_np = np.asarray(colors, dtype=np.float64)
            if colors_np.shape != (N, 3):
                splat_colors = torch.full((N, 3), 0.6, dtype=torch.float32, device=self.device)
            else:
                splat_colors = torch.clamp(torch.tensor(colors_np, dtype=torch.float32,
                                                        device=self.device), 0.0, 1.0)

        descriptors_t = None
        if include_descriptors and descriptors is not None:
            desc_np = np.asarray(descriptors, dtype=np.float64)
            if desc_np.ndim == 2 and desc_np.shape[0] == N:
                descriptors_t = torch.tensor(desc_np, dtype=torch.float32, device=self.device)

        # Splat visible features onto the (B, H, W, 3) image via index_put_.
        if torch.any(visible_mask):
            b_idx, n_idx = torch.nonzero(visible_mask, as_tuple=True)  # (K,)
            r = torch.round(v[b_idx, n_idx]).long()
            c = torch.round(u[b_idx, n_idx]).long()
            self._splat_batched(image, b_idx, r, c, sp_t[b_idx], splat_colors[n_idx])

        # Rebuild per-drone outputs from the visible mask.
        keypoints = []
        keypoint_depths = []
        visible_features = []
        visible_colors = []
        visible_descriptors = []
        for b in range(B):
            mask = visible_mask[b]
            keypoints.append(torch.stack([u[b][mask], v[b][mask]], dim=-1))
            keypoint_depths.append(depth[b][mask])
            visible_features.append(features_t[mask])
            visible_colors.append(splat_colors[mask] if include_colors else None)
            visible_descriptors.append(descriptors_t[mask] if descriptors_t is not None else None)

        colors_out = splat_colors.unsqueeze(0).expand(B, N, -1) if include_colors else None
        descriptors_out = descriptors_t.unsqueeze(0).expand(B, N, -1) if descriptors_t is not None else None

        return {
            'image': image,
            'visible_mask': visible_mask,
            'depth': depth,
            'projected': torch.stack([u, v], dim=-1),
            'keypoints': keypoints,
            'keypoint_depths': keypoint_depths,
            'visible_features': visible_features,
            'colors': colors_out,
            'visible_colors': visible_colors,
            'descriptors': descriptors_out,
            'visible_descriptors': visible_descriptors,
        }

    def _render_per_drone_worlds(self, worlds, B, image, fx, fy, cx, cy, dc, near, sp_t,
                                 R_WC, p_WC, include_colors, include_descriptors, with_distortion):
        """
        The per-drone-world render path: each drone observes its own world
        (a None entry is an empty world). Features are concatenated across the
        batch and the heavy projection math stays vectorized, but because the
        per-drone feature counts can differ the all-feature outputs are
        per-drone lists.
        """
        worlds_list = list(worlds)
        if len(worlds_list) != B:
            raise ValueError("worlds must be a single World or a sequence of length B={} (one per drone), "
                             "got {}".format(B, len(worlds_list)))

        drone_features, drone_splat_colors, drone_desc, drone_blocks, n_list = [], [], [], [], []
        for world in worlds_list:
            if world is None:
                n_list.append(0)
                drone_features.append(np.empty((0, 3), dtype=np.float64))
                drone_splat_colors.append(np.empty((0, 3), dtype=np.float32))
                drone_desc.append(None)
                drone_blocks.append([])
                continue

            features = None
            getter = getattr(world, 'get_surface_features', None)
            if getter is not None:
                features = getter()
            if features is None:
                features = np.empty((0, 3), dtype=np.float64)
            feats = np.asarray(features, dtype=np.float64).reshape(-1, 3)
            n_i = feats.shape[0]
            n_list.append(n_i)
            drone_features.append(feats)

            colors = None
            getter = getattr(world, 'get_feature_colors', None)
            if getter is None:
                getter = getattr(world, 'get_feature_descriptors', None)  # legacy worlds: RGB colors
            if getter is not None:
                colors = getter()
            colors_np = np.asarray(colors, dtype=np.float64) if colors is not None else None
            if colors_np is None or colors_np.shape != (n_i, 3):
                drone_splat_colors.append(np.full((n_i, 3), 0.6, dtype=np.float32))
            else:
                drone_splat_colors.append(np.clip(colors_np, 0.0, 1.0).astype(np.float32))

            descriptors = None
            getter = getattr(world, 'get_feature_descriptors', None)
            if getter is not None:
                descriptors = getter()
            desc_np = np.asarray(descriptors, dtype=np.float64) if descriptors is not None else None
            if desc_np is None or desc_np.ndim != 2 or desc_np.shape[0] != n_i:
                drone_desc.append(None)
            else:
                drone_desc.append(desc_np)

            drone_blocks.append(list(world.get_block_bounding_boxes()) if hasattr(world, 'get_block_bounding_boxes') else [])

        total = sum(n_list)
        empty_kp = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        empty_d = torch.empty((0,), dtype=torch.float32, device=self.device)
        empty_f = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        empty_b = torch.zeros((0,), dtype=torch.bool, device=self.device)

        if total == 0:
            return {
                'image': image,
                'visible_mask': [empty_b for _ in range(B)],
                'depth': [empty_d for _ in range(B)],
                'projected': [empty_kp for _ in range(B)],
                'keypoints': [empty_kp for _ in range(B)],
                'keypoint_depths': [empty_d for _ in range(B)],
                'visible_features': [empty_f for _ in range(B)],
                'colors': [None for _ in range(B)],
                'visible_colors': [None for _ in range(B)],
                'descriptors': [None for _ in range(B)],
                'visible_descriptors': [None for _ in range(B)],
            }

        # Concatenated features with a per-feature drone index.
        features_all = torch.tensor(np.vstack(drone_features), dtype=torch.float32, device=self.device)  # (G, 3)
        splat_colors_all = torch.tensor(np.vstack(drone_splat_colors), dtype=torch.float32,
                                        device=self.device)  # (G, 3)
        g = torch.cat([torch.full((nm,), i, dtype=torch.long, device=self.device)
                       for i, nm in enumerate(n_list)])  # (G,) per-feature drone index

        # Camera-frame points with each feature's own drone camera/pose.
        points_cam = torch.einsum('gij,gj->gi', R_WC[g], features_all - p_WC[g])  # (G, 3)
        depth = points_cam[:, 2]

        # Projection with per-feature (via per-drone) intrinsics/distortion.
        z = depth
        safe_z = torch.where(z != 0, z, torch.ones_like(z))
        x_n = points_cam[:, 0] / safe_z
        y_n = points_cam[:, 1] / safe_z
        if with_distortion:
            dcg = dc[g]
            k1, k2, p1, p2, k3 = dcg[:, 0], dcg[:, 1], dcg[:, 2], dcg[:, 3], dcg[:, 4]
            r2 = x_n**2 + y_n**2
            radial = 1.0 + k1*r2 + k2*r2**2 + k3*r2**3
            xd = x_n*radial + 2*p1*x_n*y_n + p2*(r2 + 2*x_n**2)
            yd = y_n*radial + p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n
        else:
            xd, yd = x_n, y_n
        u = fx[g] * xd + cx[g]
        v = fy[g] * yd + cy[g]

        # Occlusion per drone over its own world's blocks.
        not_occluded = torch.ones(total, dtype=torch.bool, device=self.device)
        offset = 0
        for i, world in enumerate(worlds_list):
            n_i = n_list[i]
            if n_i == 0:
                continue
            vis = self._compute_not_occluded(p_WC[i], features_all[offset:offset + n_i] - p_WC[i],
                                             drone_blocks[i], self.device)
            not_occluded[offset:offset + n_i] = vis
            offset += n_i

        visible_mask = (depth > near[g]) & (u >= 0) & (u < self.width) & \
                       (v >= 0) & (v < self.height) & not_occluded

        # Splat visible features with each drone's splat radius.
        if torch.any(visible_mask):
            idx = torch.nonzero(visible_mask)[:, 0]
            b = g[idx]
            r = torch.round(v[idx]).long()
            c = torch.round(u[idx]).long()
            self._splat_batched(image, b, r, c, sp_t[b], splat_colors_all[idx])

        # Per-drone outputs.
        keypoints, keypoint_depths, visible_features = [], [], []
        projected, visible_masks, depths, colors_out, visible_colors = [], [], [], [], []
        descriptors_out, visible_descriptors = [], []
        offset = 0
        for i in range(B):
            n_i = n_list[i]
            sl = slice(offset, offset + n_i)
            offset += n_i
            vis_b = visible_mask[sl]
            splat_b = splat_colors_all[sl]
            desc_np = drone_desc[i]
            desc_t = (torch.tensor(desc_np, dtype=torch.float32, device=self.device)
                      if desc_np is not None else None)
            visible_masks.append(vis_b)
            depths.append(depth[sl])
            projected.append(torch.stack([u[sl], v[sl]], dim=-1))
            keypoints.append(torch.stack([u[sl][vis_b], v[sl][vis_b]], dim=-1))
            keypoint_depths.append(depth[sl][vis_b])
            visible_features.append(features_all[sl][vis_b])
            colors_out.append(splat_b if include_colors else None)
            visible_colors.append(splat_b[vis_b] if include_colors else None)
            descriptors_out.append(desc_t if include_descriptors and desc_t is not None else None)
            visible_descriptors.append(desc_t[vis_b] if (include_descriptors and desc_t is not None) else None)

        return {
            'image': image,
            'visible_mask': visible_masks,
            'depth': depths,
            'projected': projected,
            'keypoints': keypoints,
            'keypoint_depths': keypoint_depths,
            'visible_features': visible_features,
            'colors': colors_out,
            'visible_colors': visible_colors,
            'descriptors': descriptors_out,
            'visible_descriptors': visible_descriptors,
        }


if __name__ == "__main__":
    import numpy as np
    from rotorpy.world import World

    cam = PinholeCamera()

    # Pure-math sanity checks (no world needed).
    pts_cam = np.array([[1.0, 0.5, 2.0]])
    print("Projected (no distortion):", cam.project_points(pts_cam))
    print("Distorted (no distortion):", cam.distort_points(pts_cam))

    # Rendering against a world (works with or without features).
    world = World.empty((0.0, 2.0, 0.0, 2.0, 0.0, 2.0), add_features=True,
                        feature_mode='regular', feature_spacing=0.5, descriptor_noise=0.0)
    state = {'x': np.array([1.0, 1.0, 1.5]), 'q': np.array([0.0, 0.0, 0.0, 1.0])}
    out = cam.render(world, state)
    print("Image shape:", out['image'].shape, "dtype:", out['image'].dtype)
    print("Total features:", out['visible_mask'].shape[0], "Visible:", int(out['visible_mask'].sum()))
    print("Keypoints:", out['keypoints'].shape)

    if torch is not None:
        batched = BatchedPinholeCamera(num_drones=2)
        states = {'x': torch.tensor([[1.0, 1.0, 1.5], [1.0, 1.0, 2.0]]),
                  'q': torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])}
        bout = batched.render(world, states)
        print("Batch image shape:", tuple(bout['image'].shape), "Visible per drone:", [int(m.sum()) for m in bout['visible_mask']])

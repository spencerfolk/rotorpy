import numpy as np
from scipy.spatial.transform import Rotation

try:
    import torch
except ImportError:
    torch = None


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

    State space:
        The vehicle state dict is expected to contain 'x' (3,) position and
        'q' (4,) orientation quaternion [i, j, k, w].
    """
    def __init__(self, intrinsics=None, extrinsics=None, near_plane=0.05, frame_rate=None,
                 splat_radius=1):
        """
        Parameters:
            intrinsics, dict of camera intrinsics, see class docstring
            extrinsics, dict of camera extrinsics, see class docstring
            near_plane, minimum camera-frame z for a feature to be rendered, m
            frame_rate, frame capture rate used by simulate(), Hz; None renders
                at every simulation step
            splat_radius, default splat radius used by render(), pixels
        """
        if frame_rate is not None and frame_rate <= 0:
            raise ValueError("frame_rate must be positive or None, got {}".format(frame_rate))
        if int(splat_radius) < 0:
            raise ValueError("splat_radius must be non-negative, got {}".format(splat_radius))

        self.frame_rate = frame_rate
        self.splat_radius = int(splat_radius)
        if intrinsics is None:
            intrinsics = {'fx': 500.0, 'fy': 500.0, 'width': 640, 'height': 480, 'cx': 320.0, 'cy': 240.0,
                          'dist_coeffs': np.array([0.0, 0.0, 0.0, 0.0, 0.0])}  # [k1, k2, p1, p2, k3]
        if extrinsics is None:
            extrinsics = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.0, 0.0, 0.0, 1.0])}

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.near_plane = near_plane

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
        pattern used by other RotorPy sensors (IMU, mocap, range sensors).

        Inputs:
            state, a dict describing the vehicle state with keys
                x, position, m, shape=(3,)
                q, orientation quaternion [i, j, k, w], shape=(4,)
            world, World object exposing get_surface_features(),
                get_feature_descriptors(), and get_block_bounding_boxes()
            **render_kwargs, additional keyword arguments forwarded to render()
                (e.g. background_color, splat_radius, with_distortion)

        Outputs:
            measurement, a dict with keys
                image, (H, W, 3) float32 image in [0, 1]
                keypoints, (M, 2) pixels of visible in-frustum features
                keypoint_depths, (M,) camera-frame z of visible features
                visible_features, (M, 3) world positions of visible features
                visible_mask, (N,) bool over all world features
                projected, (N, 2) pixels of all features (may be out of bounds)
                depth, (N,) camera-frame z of all features
        """
        return self.render(world, state, **render_kwargs)

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

    def render(self, world, state, background_color=None, splat_radius=None, with_distortion=True):
        """
        Render a synthetic image of the world's surface features.

        Inputs:
            world, World object exposing get_surface_features(),
                get_feature_descriptors(), and get_block_bounding_boxes()
            state, a dict describing the vehicle state with keys
                x, position, m, shape=(3,)
                q, orientation quaternion [i, j, k, w], shape=(4,)
            background_color, RGB tuple in [0, 1] (default [0.9, 0.9, 0.9])
            splat_radius, features are splatted on a (2*splat_radius+1) square
                patch centered on their rounded pixel; None uses the camera's
                instance default (see __init__)
            with_distortion, if True, apply the distortion model

        Outputs:
            render, a dict with keys
                image, (H, W, 3) float32 image in [0, 1]
                keypoints, (M, 2) pixels of visible in-frustum features
                keypoint_depths, (M,) camera-frame z of visible features
                visible_features, (M, 3) world positions of visible features
                visible_mask, (N,) bool over all world features
                projected, (N, 2) pixels of all features (may be out of bounds)
                depth, (N,) camera-frame z of all features
        """
        if splat_radius is None:
            splat_radius = self.splat_radius
        if background_color is None:
            background_color = [0.9, 0.9, 0.9]
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
            }

        points_cam = self.world_to_camera(features, camera_pose)
        depth = points_cam[:, 2]
        projected = self.project_points(points_cam, with_distortion=with_distortion)
        u = projected[:, 0]
        v = projected[:, 1]
        not_occluded = self.compute_occlusion(features, camera_pose, world)

        visible_mask = (depth > self.near_plane) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height) & not_occluded

        if descriptors is None:
            colors = np.full((N, 3), 0.6, dtype=np.float64)
        else:
            colors = np.clip(np.asarray(descriptors, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
            if colors.shape[0] != N:
                colors = np.full((N, 3), 0.6, dtype=np.float64)

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
            values = np.broadcast_to(colors[visible_idx].astype(np.float32)[:, None, None, :],
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
        }


class BatchedPinholeCamera:
    """
    Torch-parallelized version of PinholeCamera that renders the same world
    for a batch of drones simultaneously.

    Parameters:
        num_drones, number of drones in the batch (the actual batch size B
            passed to render() may be <= num_drones)
        intrinsics, dict of camera intrinsics, see PinholeCamera
        extrinsics, dict of camera extrinsics, see PinholeCamera
        near_plane, minimum camera-frame z for a feature to be rendered, m
        device, torch device string, e.g. 'cpu' or 'cuda'
    """
    def __init__(self, num_drones, intrinsics=None, extrinsics=None, near_plane=0.05, device='cpu'):
        """
        Parameters:
            num_drones, number of drones in the batch
            intrinsics, dict of camera intrinsics, see PinholeCamera
            extrinsics, dict of camera extrinsics, see PinholeCamera
            near_plane, minimum camera-frame z for a feature to be rendered, m
            device, torch device string
        """
        if torch is None:
            raise ImportError("torch required for BatchedPinholeCamera")

        self.num_drones = num_drones
        self.device = torch.device(device)
        self.near_plane = near_plane

        if intrinsics is None:
            intrinsics = {'fx': 500.0, 'fy': 500.0, 'width': 640, 'height': 480, 'cx': 320.0, 'cy': 240.0,
                          'dist_coeffs': np.array([0.0, 0.0, 0.0, 0.0, 0.0])}  # [k1, k2, p1, p2, k3]
        if extrinsics is None:
            extrinsics = {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.0, 0.0, 0.0, 1.0])}

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.fx = float(intrinsics['fx'])
        self.fy = float(intrinsics['fy'])
        self.width = int(intrinsics['width'])
        self.height = int(intrinsics['height'])
        self.cx = float(intrinsics['cx'])
        self.cy = float(intrinsics['cy'])

        p_BC = np.asarray(extrinsics['position'], dtype=np.float64)
        q_BC = np.asarray(extrinsics['orientation'], dtype=np.float64)
        if p_BC.shape != (3,):
            raise ValueError("extrinsics['position'] must have shape (3,), got {}".format(p_BC.shape))
        if q_BC.shape != (4,):
            raise ValueError("extrinsics['orientation'] must have shape (4,), got {}".format(q_BC.shape))
        self.p_BC = torch.tensor(p_BC, dtype=torch.float32, device=self.device)
        self.R_BC = torch.tensor(Rotation.from_quat(q_BC).as_matrix(), dtype=torch.float32, device=self.device)  # body -> camera

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

    def measurement(self, world, states, **render_kwargs):
        """
        Compute camera measurements for a batch of drones.

        This is the primary sensor interface, consistent with the measurement()
        pattern used by other RotorPy sensors (IMU, mocap, range sensors).

        Inputs:
            world, World object exposing get_surface_features(),
                get_feature_descriptors(), and get_block_bounding_boxes()
            states, a dict describing the vehicle states with keys
                x, position, (B, 3) tensor
                q, orientation quaternion [i, j, k, w], (B, 4) tensor
            **render_kwargs, additional keyword arguments forwarded to render()
                (e.g. background_color, splat_radius, with_distortion)

        Outputs:
            measurement, a dict with keys
                image, (B, H, W, 3) float tensor in [0, 1]
                visible_mask, (B, N) bool tensor over all world features
                depth, (B, N) camera-frame z tensor of all features
                keypoints, list of length B of (M_b, 2) pixel tensors
                keypoint_depths, list of length B of (M_b,) depth tensors
                visible_features, list of length B of (M_b, 3) world-position tensors
        """
        return self.render(world, states, **render_kwargs)

    def render(self, world, states, background_color=None, splat_radius=1, with_distortion=True):
        """
        Render synthetic images for a batch of drones.

        Inputs:
            world, World object exposing get_surface_features(),
                get_feature_descriptors(), and get_block_bounding_boxes()
            states, a dict describing the vehicle states with keys
                x, position, (B, 3) tensor
                q, orientation quaternion [i, j, k, w], (B, 4) tensor
            background_color, RGB tuple in [0, 1] (default [0.9, 0.9, 0.9])
            splat_radius, features are splatted on a (2*splat_radius+1) square
                patch centered on their rounded pixel
            with_distortion, if True, apply the distortion model

        Outputs:
            render, a dict with keys
                image, (B, H, W, 3) float tensor in [0, 1]
                visible_mask, (B, N) bool tensor over all world features
                depth, (B, N) camera-frame z tensor of all features
                keypoints, list of length B of (M_b, 2) pixel tensors
                keypoint_depths, list of length B of (M_b,) depth tensors
                visible_features, list of length B of (M_b, 3) world-position tensors
        """
        if background_color is None:
            background_color = [0.9, 0.9, 0.9]

        x = torch.as_tensor(states['x']).float().to(self.device)
        q = torch.as_tensor(states['q']).float().to(self.device)
        B = x.shape[0]
        if B > self.num_drones:
            raise ValueError("batch size B={} exceeds num_drones={}".format(B, self.num_drones))

        features = None
        getter = getattr(world, 'get_surface_features', None)
        if getter is not None:
            features = getter()
        if features is None:
            features = np.empty((0, 3), dtype=np.float64)

        descriptors = None
        getter = getattr(world, 'get_feature_descriptors', None)
        if getter is not None:
            descriptors = getter()

        features_np = np.asarray(features, dtype=np.float64).reshape(-1, 3)
        N = features_np.shape[0]
        features_t = torch.tensor(features_np, dtype=torch.float32, device=self.device)

        image = torch.full((B, self.height, self.width, 3), float(background_color[0]),
                           dtype=torch.float32, device=self.device)
        image[:, :, :, 0] = background_color[0]
        image[:, :, :, 1] = background_color[1]
        image[:, :, :, 2] = background_color[2]

        empty_kp = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        empty_d = torch.empty((0,), dtype=torch.float32, device=self.device)
        empty_f = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        if N == 0:
            return {
                'image': image,
                'visible_mask': torch.zeros((B, 0), dtype=torch.bool, device=self.device),
                'depth': torch.zeros((B, 0), dtype=torch.float32, device=self.device),
                'keypoints': [empty_kp for _ in range(B)],
                'keypoint_depths': [empty_d for _ in range(B)],
                'visible_features': [empty_f for _ in range(B)],
            }

        # World-to-camera rotation per drone: R_WC = R_BC @ R_WB^T.
        R_WB = self._quat_to_rotmat(q)  # (B, 3, 3), body -> world
        R_WC = torch.einsum('ij,bjk->bik', self.R_BC, R_WB.transpose(-1, -2))  # (B, 3, 3)

        # Camera world position per drone: p_WC = x + R_WB @ p_BC.
        p_WC = x + torch.einsum('bij,j->bi', R_WB, self.p_BC)  # (B, 3)

        # Camera-frame points: p_c = R_WC @ (p_W - p_WC).
        points_cam = torch.einsum('bij,bnj->bni', R_WC, features_t - p_WC.unsqueeze(1))  # (B, N, 3)
        depth = points_cam[:, :, 2]  # (B, N)

        # Projection.
        z = points_cam[:, :, 2]
        safe_z = torch.where(z != 0, z, torch.ones_like(z))
        x_n = points_cam[:, :, 0] / safe_z
        y_n = points_cam[:, :, 1] / safe_z
        if with_distortion:
            k1, k2, p1, p2, k3 = [float(v) for v in self.intrinsics['dist_coeffs']]
            r2 = x_n**2 + y_n**2
            radial = 1.0 + k1*r2 + k2*r2**2 + k3*r2**3
            xd = x_n*radial + 2*p1*x_n*y_n + p2*(r2 + 2*x_n**2)
            yd = y_n*radial + p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n
        else:
            xd, yd = x_n, y_n
        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy

        # Occlusion (loop over blocks, vectorized over (B, N)).
        not_occluded = torch.ones((B, N), dtype=torch.bool, device=self.device)
        origin = p_WC  # (B, 3)
        direction = features_t.unsqueeze(0) - origin.unsqueeze(1)  # (B, N, 3)
        for extents in world.get_block_bounding_boxes():
            box = np.asarray(extents, dtype=np.float64)
            bmin = torch.tensor(box[[0, 2, 4]], dtype=torch.float32, device=self.device)  # (3,)
            bmax = torch.tensor(box[[1, 3, 5]], dtype=torch.float32, device=self.device)  # (3,)
            d_mask = direction != 0.0
            safe_d = torch.where(d_mask, direction, torch.ones_like(direction))
            t1 = (bmin.unsqueeze(0).unsqueeze(0) - origin.unsqueeze(1)) / safe_d  # (B, N, 3)
            t2 = (bmax.unsqueeze(0).unsqueeze(0) - origin.unsqueeze(1)) / safe_d
            in_slab = (origin.unsqueeze(1) >= bmin) & (origin.unsqueeze(1) <= bmax)  # (B, 1, 3)
            inf = torch.tensor(float('inf'), dtype=torch.float32, device=self.device)
            t1 = torch.where(d_mask, t1, torch.where(in_slab, -inf, inf))
            t2 = torch.where(d_mask, t2, torch.where(in_slab, inf, inf))
            tmin = torch.minimum(t1, t2)
            tmax = torch.maximum(t1, t2)
            t_entry = torch.max(tmin, dim=-1).values
            t_exit = torch.min(tmax, dim=-1).values
            hit = (t_entry <= t_exit) & (t_entry < 1 - 1e-6) & (t_exit > 1e-6)
            not_occluded &= ~hit

        visible_mask = (depth > self.near_plane) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height) & not_occluded

        if descriptors is None:
            colors = torch.full((N, 3), 0.6, dtype=torch.float32, device=self.device)
        else:
            colors = torch.clamp(torch.tensor(np.asarray(descriptors, dtype=np.float64).reshape(-1, 3),
                                              dtype=torch.float32, device=self.device), 0.0, 1.0)
            if colors.shape[0] != N:
                colors = torch.full((N, 3), 0.6, dtype=torch.float32, device=self.device)

        # Splat visible features onto the (B, H*W, 3) image via index_put_.
        if torch.any(visible_mask):
            b_idx, n_idx = torch.nonzero(visible_mask, as_tuple=True)  # (K,)
            r = torch.round(v[b_idx, n_idx]).long()
            c = torch.round(u[b_idx, n_idx]).long()
            color_src = colors[n_idx]  # (K, 3)
            dr = torch.arange(-splat_radius, splat_radius + 1, device=self.device)
            dc = torch.arange(-splat_radius, splat_radius + 1, device=self.device)
            rows = r[:, None, None] + dr[None, :, None]  # (K, Pr, 1)
            cols = c[:, None, None] + dc[None, None, :]  # (K, 1, Pc)
            inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
            rows = rows.clamp(0, self.height - 1)
            cols = cols.clamp(0, self.width - 1)
            full_idx = b_idx[:, None, None] * (self.height * self.width) + rows * self.width + cols  # (K, Pr, Pc)
            full_idx = full_idx[inside]
            values = color_src[:, None, None, :].expand(-1, rows.shape[1], cols.shape[2], -1)[inside]
            # image is freshly allocated and contiguous, so reshape returns a
            # view and index_put_ writes through to it directly.
            image.reshape(B * self.height * self.width, 3).index_put_((full_idx,), values)

        # Rebuild per-drone outputs from the visible mask.
        keypoints = []
        keypoint_depths = []
        visible_features = []
        for b in range(B):
            mask = visible_mask[b]
            keypoints.append(torch.stack([u[b][mask], v[b][mask]], dim=-1))
            keypoint_depths.append(depth[b][mask])
            visible_features.append(features_t[mask])

        return {
            'image': image,
            'visible_mask': visible_mask,
            'depth': depth,
            'keypoints': keypoints,
            'keypoint_depths': keypoint_depths,
            'visible_features': visible_features,
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

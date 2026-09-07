# World Module

Worlds are used to represent obstacles in the map. Maps are implemented as `.json` files using the following structure: 

The world bounds are defined by the "bounds" key: 
```
{
    "bounds": {"extents": [xmin, xmax, ymin, ymax, zmin, zmax]},

```
Obstacles are defined using cuboids. The cuboids are defined as follows: 
```
    "blocks": [
        {"extents": [xmin, xmax, ymin, ymax, zmin, zmax], "color": [R, G, B]},
        ...
    ]
}
```
where the "color" value is a specified by RGB values between 0 and 1. As an example: 
```
{"extents": [0, 1, -3, 2, 0, 10], "color": [0, 1, 0]}
```
will create a box with corners (0,-3,0), (0, 2,0), (0,-3,10), (0,2,10), (1,-3,0), (1,2,0), (1,-3,10), (1,2,10). In other words, a box that has LWH dimensions (1m)x(5m)x(10m). The `[0,1,0]` label indicates it will be rendered in green. 

## Visual features

Worlds can carry visual features for the camera sensor (see `rotorpy/sensors/camera.py`). Each feature is a 3D world point with an RGB `color` in [0, 1], and may additionally carry a generic descriptor vector (`descriptors`, any dimension, e.g. 128-d SIFT or 96-d ALIKED) for descriptor-based visual odometry or vision policies. Features are either generated from the blocks by a named generator, or embedded verbatim in the JSON for exact reproducibility.

### Generated features

A world file with a `features` section automatically generates features on load (it takes precedence over the `World(...)` constructor's `feature_mode`/`add_features` arguments). The section configures a generator by `mode`:

```
{
    "bounds": {"extents": [...]},
    "blocks": [...],
    "features": {"mode": "regular", "feature_spacing": 0.2, "descriptor_noise": 0.15, "descriptor_dim": 128, "seed": 42}
}
```

Available modes and their parameters:

| Mode | Placement | Parameters |
| --- | --- | --- |
| `regular` | uniform grid over all block faces | `feature_spacing` (m) |
| `random` | random splatter at areal density | `feature_density` (features per m^2) |
| `edge_uniform` | uniform spacing along all 12 block edges | `edge_spacing` (m) |
| `edge_random` | random features along block edges | `edge_density` (features per m) |

All modes accept `descriptor_noise` (standard deviation of the Gaussian jitter added to each block's color to form per-feature RGB, default 0.05), an optional `descriptor_dim` that attaches L2-normalized synthetic descriptor vectors of that dimension to every feature, and an optional `seed` for reproducible generation.

### Embedded features

Instead of a generator, explicit features can be embedded directly (no generator and no RNG involved, so results are exact across trials and reloads):

```
"features": {
    "points": [[0.1, 0.2, 0.0], [0.3, 0.4, 0.1], ...],
    "colors": [[0.9, 0.2, 0.2], ...],        // optional, defaults to grey [0.6, 0.6, 0.6]
    "descriptors": [[...], ...],             // optional (N, D) descriptor vectors, any dimension D
    "descriptor_type": "sift",               // optional label for the descriptor family
    "blocks": [0, 0, 1, ...]                 // optional per-feature block index, defaults to -1
}
```

`World.to_file('filename.json', include_features=True)` writes such an embedded section for the current world. 
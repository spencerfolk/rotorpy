''' 
Test utilities that are commonly used in RotorPy. 
''' 

import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
import copy
import json

# Create a test world data dictionary. 
test_world_data = {
    'bounds': {'extents': [0, 5.0, 0, 2.0, 0, 13.0]},
    'blocks': [
        {'extents': [2, 3, 0.0, 2, 0.0, 10.0], 'color': [1, 0, 0]},
        {'extents': [2, 3, 0.0, 2, 0.0, 10.0], 'color': [1, 0, 0]},
        {'extents': [2, 3, 0.0, 2, 0.0, 10.0], 'color': [1, 0, 0]},
        {'extents': [2, 3, 0.0, 2, 0.0, 10.0], 'color': [1, 0, 0]},
        {'extents': [2, 3, 0.0, 2, 0.0, 10.0]},
        {'extents': [2, 3, 0.0, 2, 0.0, 10.0]},
        {'extents': [2, 3, 0.0, 2, 0.0, 10.0]}],
    'start': np.array([0, 0, 1]),
    'goal': np.array([4, 0, 2]),
    'resolution': np.array([0.25, 0.25, 0.5]),
    'margin': 0.1,
    'expected_path_length': 20.52
    }

def test_animate():
    from rotorpy.utils.animate import animate
    from rotorpy.world import World
    
    print("\nTesting animate() function")
    
    # Create dummy data. 
    world = World.empty([-1, 1, -1, 1, -1, 1])
    T = np.linspace(0, 1, 100)
    position = np.zeros((T.size, 3))
    rotation = np.eye(3).reshape((1, 3, 3)).repeat(T.size, axis=0)
    wind = np.zeros((T.size, 3))
    
    # Animate the results.
    ani = animate(T, position, rotation, wind=wind, animate_wind=False, world=world, filename=None, blit=False, show_axes=True, close_on_finish=True)

def test_numpy_encoding():
    from rotorpy.utils.numpy_encoding import NumpyJSONEncoder, to_ndarray

    print("\nTesting numpy_encoding.py")

    # Retrieve example data.
    data = copy.deepcopy(test_world_data)

    # Write to a JSON file.
    with open('test_utils.json', 'w') as file:
        file.write(json.dumps(data, cls=NumpyJSONEncoder, indent=4))

    # Read from the JSON file.
    with open('test_utils.json') as file:
        data_out = json.load(file)
    data_out = to_ndarray(data_out)

    # Delete the JSON file.
    os.remove('test_utils.json')

    # Ensure that data_out is the same as data. 
    assert np.allclose(data['bounds']['extents'], data_out['bounds']['extents']), "NumPy array encoding failed."
    assert np.allclose(data['start'], data_out['start']), "NumPy array encoding failed."
    assert np.allclose(data['goal'], data_out['goal']), "NumPy array encoding failed."
    assert np.allclose(data['resolution'], data_out['resolution']), "NumPy array encoding failed."
    assert data['margin'] == data_out['margin'], "NumPy array encoding failed."
    assert data['expected_path_length'] == data_out['expected_path_length'], "NumPy array encoding failed."

def test_occupancy_map():
    from rotorpy.world import World
    from rotorpy.utils.occupancy_map import OccupancyMap

    print("\nTesting occupancy_map.py")

    # Load test world
    world = World(test_world_data)

    # Create a figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Draw the world
    world.draw(ax)

    # Create an Occupancy map
    oc = OccupancyMap(world, (.5, .5, .5), .01)
    position = np.array([2.5, 2.5, 0.5])
    region = (1, 1)
    oc.get_local_2d_occupancy_map(position, region_size=region)

    # Draw the occupancy map (may be slow for many voxels; will look weird if plotted on top of a world.draw)
    oc.draw(ax)

def test_plotter():
    from rotorpy.utils.animate import animate
    from rotorpy.utils.plotter import Plotter, plot_map
    from rotorpy.world import World

    print("\nTesting plotter.py")

    # Load the test world data.
    world = World(test_world_data)
    
    # Plot the world. 
    (fig, ax) = plt.subplots(nrows=1, ncols=1, num="Top Down World View")
    plot_map(ax, world.world)
    ax.set_title("Test World Plotting")

def test_shapes():
    from rotorpy.utils.shapes import Face, Cuboid, Cylinder, Quadrotor

    print("\nTesting shapes.py")

    # Test Face
    fig = plt.figure(num=4, clear=True)
    ax = fig.add_subplot(projection='3d')
    corners = np.array([(1,1,1), (-1,1,1), (-1,-1,1), (1,-1,1)])
    z_plus_face = Face(ax, corners=corners, facecolors='b')
    x_plus_face = Face(ax, corners=corners, facecolors='r')
    x_plus_face.transform(
        position=(0,0,0),
        rotation=Rotation.from_rotvec(np.pi/2 * np.array([0, 1, 0])).as_matrix())
    y_plus_face = Face(ax, corners=corners, facecolors='g')
    y_plus_face.transform(
        position=(0,0,0),
        rotation=Rotation.from_rotvec(np.pi/2 * np.array([-1, 0, 0])).as_matrix())
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Test Cuboid
    fig = plt.figure(num=0, clear=True)
    ax = fig.add_subplot(projection='3d')
    cuboid = Cuboid(ax, x_span=1, y_span=1, z_span=1)
    cuboid.transform(position=np.array([[0, 0, 0]]), rotation=np.identity(3))
    rotation = Rotation.from_rotvec(np.pi/4 * np.array([1, 0, 0])).as_matrix()
    cuboid = Cuboid(ax, x_span=1, y_span=2, z_span=3)
    cuboid.transform(position=np.array([[2.0, 1, 1]]), rotation=rotation)
    ax.set_xlim(-1,3)
    ax.set_ylim(-1,3)
    ax.set_zlim(-1,3)

    # Test Cylinder
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot(projection='3d')
    cylinder = Cylinder(ax, radius=0.2, height=0.2)
    cylinder.transform(position=np.array([[0, 0, 0]]), rotation=np.identity(3))
    rotation = Rotation.from_rotvec(np.pi/4 * np.array([1, 0, 0])).as_matrix()
    cylinder = Cylinder(ax, radius=0.2, height=0.2)
    cylinder.transform(position=np.array([[1.0, 0, 0]]), rotation=rotation)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    # Test Quadrotor
    fig = plt.figure(num=2, clear=True)
    ax = fig.add_subplot(projection='3d')
    quad = Quadrotor(ax)
    quad.transform(position=np.array([[0.5, 0.5, 0.5]]), rotation=np.identity(3))
    quad = Quadrotor(ax)
    rotation = Rotation.from_rotvec(np.pi/4 * np.array([1, 0, 0])).as_matrix()
    quad.transform(position=np.array([[0.8, 0.8, 0.8]]), rotation=rotation)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    # Test Cuboid coloring.
    fig = plt.figure(num=3, clear=True)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-3.25,3.25)
    ax.set_ylim(-3.25,3.25)
    ax.set_zlim(-3.25,3.25)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # No shading.
    x = (-1, 0, 1)
    z = -3.25
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, shade=False)
    cuboid.transform(position=(x[0], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, shade=False)
    cuboid.transform(position=(x[1], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, shade=False, facecolors='b')
    cuboid.transform(position=(x[2], 0, z))

    # Shading.
    z = z + 1
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5)
    cuboid.transform(position=(x[0], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5)
    cuboid.transform(position=(x[1], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, facecolors='b')
    cuboid.transform(position=(x[2], 0, z))

    # Transparency.
    z = z + 1
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, alpha=0.5)
    cuboid.transform(position=(x[0], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, alpha=0.5)
    cuboid.transform(position=(x[1], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, alpha=0.5, facecolors='b')
    cuboid.transform(position=(x[2], 0, z))

    # No shading, edge lines.
    z = z + 1
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, shade=False, linewidth=1)
    cuboid.transform(position=(x[0], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, shade=False, linewidth=1, edgecolors='k')
    cuboid.transform(position=(x[1], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, shade=False, linewidth=1, facecolors='b', edgecolors='k')
    cuboid.transform(position=(x[2], 0, z))

    # Shading, edge lines.
    z = z + 1
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, linewidth=1)
    cuboid.transform(position=(x[0], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, linewidth=1, edgecolors='k')
    cuboid.transform(position=(x[1], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, linewidth=1, facecolors='b', edgecolors='k')
    cuboid.transform(position=(x[2], 0, z))

    # Transparency, edge lines.
    z = z + 1
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, alpha=0.5, linewidth=1)
    cuboid.transform(position=(x[0], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, alpha=0.5, linewidth=1, edgecolors='k')
    cuboid.transform(position=(x[1], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, alpha=0.5, linewidth=1, facecolors='b', edgecolors='k')
    cuboid.transform(position=(x[2], 0, z))

    # Transparent edges.
    z = z + 1
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, alpha=0, linewidth=1, edgecolors='k', antialiased=False)
    cuboid.transform(position=(x[1], 0, z))
    cuboid = Cuboid(ax, x_span=0.5, y_span=0.5, z_span=0.5, alpha=0, linewidth=1, edgecolors='k', antialiased=True)
    cuboid.transform(position=(x[2], 0, z))
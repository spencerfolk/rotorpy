"""
Parametric 3D shapes for spatial plots and animations. Shapes are drawn on an
Axes3D axes, and then can be moved using .transform(). They can return a list of
artists to support blitting in animations.

TODO:
  There is a fair amount of code duplication here; a superclass may be warranted.

"""

import itertools
import numpy as np
from mpl_toolkits.mplot3d import art3d
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation

class Face():

    def __init__(self, ax, corners, *,
        shade=True,
        alpha=1.0,
        facecolors=None,
        edgecolors=None,
        linewidth=0,
        antialiased=True):
        """
        Parameters
            ax, Axes3D to contain new shape
            corners, shape=(N,3)
            shade, shade faces using default lightsource, default is True
            linewidth, width of lines, default is 0
            alpha, transparency value in domain [0,1], default is 1.0
            edgecolors, color of edges
            facecolors, color of faces
            antialiased, smoother edge lines, default is True
        """
        self.shade = shade
        self.facecolors = facecolors

        self.ax = ax

        if self.facecolors is None:
            self.facecolors = self.ax._get_lines.get_next_color()
        self.facecolors = np.array(mcolors.to_rgba(self.facecolors))

        # Precompute verticies and normal vectors in reference configuration.
        self.verts = np.reshape(corners, (1, -1, 3))
        self.normals = np.asarray(self.ax._generate_normals(self.verts))

        # Instantiate and add collection.
        self.polyc = art3d.Poly3DCollection(self.verts, linewidth=linewidth, antialiased=antialiased, alpha=alpha, edgecolors=edgecolors, facecolors=self.facecolors)
        self.artists = (self.polyc,)
        self.transform(np.zeros((3,)), np.identity(3))
        self.ax.add_collection(self.polyc)

    def transform(self, position, rotation=np.identity(3)):

        position = np.array(position)
        position.shape = (3,1)

        # The verts array is indexed as (i_face, j_coordinate, k_point).
        verts = np.swapaxes(self.verts, 1, 2)
        new_verts = np.matmul(rotation, verts) + position
        self.polyc.set_verts(np.swapaxes(new_verts, 1, 2))

        if self.shade:
            normals = np.matmul(rotation, self.normals.T).T
            colset = self.ax._shade_colors(self.facecolors, normals)
        else:
            colset = self.facecolors
        self.polyc.set_facecolors(colset)

class Cuboid():

    def __init__(self, ax, x_span, y_span, z_span, *,
        shade=True,
        alpha=1.0,
        facecolors=None,
        edgecolors=None,
        linewidth=0,
        antialiased=True):
        """
        Parameters
            ax, Axes3D to contain new shape
            x_span, width in x-direction
            y_span, width in y-direction
            z_span, width in z-direction
            shade, shade faces using default lightsource, default is True
            linewidth, width of lines, default is 0
            alpha, transparency value in domain [0,1], default is 1.0
            edgecolors, color of edges
            facecolors, color of faces
            antialiased, smoother edge lines, default is True
        """
        self.shade = shade
        self.facecolors = facecolors

        self.ax = ax

        if self.facecolors is None:
            self.facecolors = self.ax._get_lines.get_next_color()
        self.facecolors = np.array(mcolors.to_rgba(self.facecolors))

        # Precompute verticies and normal vectors in reference configuration.
        self.verts = self.build_verts(x_span, y_span, z_span)
        self.normals = np.asarray(self.ax._generate_normals(self.verts))

        # Instantiate and add collection.
        self.polyc = art3d.Poly3DCollection(self.verts, linewidth=linewidth, antialiased=antialiased, alpha=alpha, edgecolors=edgecolors, facecolors=self.facecolors)
        self.artists = (self.polyc,)
        self.transform(np.zeros((3,)), np.identity(3))
        self.ax.add_collection(self.polyc)

    def transform(self, position, rotation=np.identity(3)):

        position = np.array(position)
        position.shape = (3,1)

        # The verts array is indexed as (i_face, j_coordinate, k_point).
        verts = np.swapaxes(self.verts, 1, 2)
        new_verts = np.matmul(rotation, verts) + position
        self.polyc.set_verts(np.swapaxes(new_verts, 1, 2))

        if self.shade:
            normals = np.matmul(rotation, self.normals.T).T
            colset = self.ax._shade_colors(self.facecolors, normals)
        else:
            colset = self.facecolors
        self.polyc.set_facecolors(colset)

    def build_verts(self, x_span, y_span, z_span):
        """
        Input
            x_span, width in x-direction
            y_span, width in y-direction
            z_span, width in z-direction
        Returns
            verts, shape=(6_faces, 4_points, 3_coordinates)
        """

        # Coordinates of each point.
        (x, y, z) = (x_span, y_span, z_span)
        bot_pts = np.array([
            [0, 0, 0],
            [x, 0, 0],
            [x, y, 0],
            [0, y, 0]])
        top_pts = np.array([
            [0, 0, z],
            [x, 0, z],
            [x, y, z],
            [0, y, z]])
        pts = np.concatenate((bot_pts, top_pts), axis=0)

        # Indices of points for each face.
        side_faces = [(i, (i+1)%4, 4+((i+1)%4), 4+i) for i in range(4)]
        side_faces = np.array(side_faces, dtype=int)
        bot_faces = np.arange(4, dtype=int)
        bot_faces.shape = (1,4)
        top_faces = 4 + bot_faces
        all_faces = np.concatenate((side_faces, bot_faces, top_faces), axis=0)

        # Vertex list.
        xt = pts[:,0][all_faces]
        yt = pts[:,1][all_faces]
        zt = pts[:,2][all_faces]
        verts = np.stack((xt, yt, zt), axis=-1)

        return verts

class Cylinder():

    def __init__(self, ax, radius, height, n_pts=8, shade=True, color=None):
        self.shade = shade

        self.ax = ax

        if color is None:
            color = self.ax._get_lines.get_next_color()
        self.color = np.array(mcolors.to_rgba(color))

        # Precompute verticies and normal vectors in reference configuration.
        self.verts = self.build_verts(radius, height, n_pts)
        self.normals = np.asarray(self.ax._generate_normals(self.verts))

        # Instantiate and add collection.
        self.polyc = art3d.Poly3DCollection(self.verts, color='b', linewidth=0, antialiased=False)
        self.artists = (self.polyc,)
        self.transform(np.zeros((3,)), np.identity(3))
        self.ax.add_collection(self.polyc)

    def transform(self, position, rotation):

        position.shape = (3,1)

        # The verts array is indexed as (i_triangle, j_coordinate, k_point).
        verts = np.swapaxes(self.verts, 1, 2)
        new_verts = np.matmul(rotation, verts) + position
        self.polyc.set_verts(np.swapaxes(new_verts, 1, 2))

        if self.shade:
            normals = np.matmul(rotation, self.normals.T).T
            colset = self.ax._shade_colors(self.color, normals)
        else:
            colset = self.color
        self.polyc.set_facecolors(colset)

    def build_verts(self, radius, height, n_pts):
        """
        Input
            radius, radius of cylinder
            height, height of cylinder
            n_pts, number of points used to describe rim of cylinder
        Returns
            verts, [n_triangles, 3_points, 3_coordinates]
        """

        theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        delta_theta = (theta[1]-theta[0])/2

        # Points around the bottom rim, top rim, bottom center, and top center.
        bot_pts = np.zeros((3, n_pts))
        bot_pts[0,:] = radius * np.cos(theta)
        bot_pts[1,:] = radius * np.sin(theta)
        bot_pts[2,:] = np.full(n_pts, -height/2)
        top_pts = np.zeros((3, n_pts))
        top_pts[0,:] = radius * np.cos(theta + delta_theta)
        top_pts[1,:] = radius * np.sin(theta + delta_theta)
        top_pts[2,:] = np.full(n_pts, height/2)
        bot_center = np.array([[0], [0], [-height/2]])
        top_center = np.array([[0], [0],  [height/2]])
        pts = np.concatenate((bot_pts, top_pts, bot_center, top_center), axis=1)

        # Triangle indices for the shell.
        up_triangles = np.stack((
            np.arange(0, n_pts, dtype=int),
            np.arange(1, n_pts+1, dtype=int),
            np.arange(n_pts+0, n_pts+n_pts, dtype=int)))
        up_triangles[1,-1] = 0
        down_triangles = np.stack((
            np.arange(0, n_pts, dtype=int),
            np.arange(n_pts, n_pts+n_pts, dtype=int),
            np.arange(n_pts-1, n_pts+n_pts-1, dtype=int)))
        down_triangles[2,0] = n_pts+n_pts-1
        shell_triangles = np.concatenate((up_triangles, down_triangles), axis=1)

        # Triangle indices for the bottom.
        bot_triangles = np.stack((
            np.arange(0, n_pts, dtype=int),
            np.arange(1, n_pts+1, dtype=int),
            np.full(n_pts, 2*n_pts, dtype=int)))
        bot_triangles[1,-1] = 0
        top_triangles = np.stack((
            np.arange(n_pts+0, n_pts+n_pts, dtype=int),
            np.arange(n_pts+1, n_pts+n_pts+1, dtype=int),
            np.full(n_pts, 2*n_pts+1, dtype=int)))
        top_triangles[1,-1] = n_pts

        all_triangles = np.concatenate((shell_triangles, bot_triangles, top_triangles), axis=1)

        xt = pts[0,:][all_triangles.T]
        yt = pts[1,:][all_triangles.T]
        zt = pts[2,:][all_triangles.T]
        verts = np.stack((xt, yt, zt), axis=-1)

        return verts

class Quadrotor():

    def __init__(self, ax,
        arm_length=0.125, rotor_radius=0.08, n_rotors=4,
        shade=True, color=None, wind=True):

        self.ax = ax
        self.wind_bool = wind

        # Apply same color to all rotor objects.
        if color is None:
            color = self.ax._get_lines.get_next_color()
        self.color = np.array(mcolors.to_rgba(color))

        # Precompute positions and rotations in the reference configuration.
        theta = np.linspace(0, 2*np.pi, n_rotors, endpoint=False)
        theta = theta + np.mean(theta[:2])
        self.rotor_position = np.zeros((3, n_rotors))
        self.rotor_position[0,:] = arm_length*np.cos(theta)
        self.rotor_position[1,:] = arm_length*np.sin(theta)

        # Instantiate.
        self.rotors = [Cylinder(ax,
                                rotor_radius,
                                0.1*rotor_radius,
                                shade=shade,
                                color=color) for _ in range(n_rotors)]
        artists = [r.artists for r in self.rotors]
        if self.wind_bool:
            self.wind_vector = [self.ax.quiver(0,0,0,0,0,0, color='k')]
            artists.append(self.wind_vector)
        self.artists = tuple(itertools.chain.from_iterable(artists))
                             
        self.transform(np.zeros((3,)), np.identity(3), np.zeros((3,)))

    def transform(self, position, rotation, wind=np.array([1,0,0])):
        position.shape = (3,1)
        wind.shape = (3,1)
        for (r, pos) in zip(self.rotors, self.rotor_position.T):
            pos.shape = (3,1)
            r.transform(np.matmul(rotation,pos)+position, rotation)
        if self.wind_bool:
            self.wind_vector[0].remove()
            self.wind_vector = [self.ax.quiver(position[0], position[1], position[2], wind[0], wind[1], wind[2], color='r', linewidth=1.5)]

if __name__ == '__main__':
    from axes3ds import Axes3Ds
    import matplotlib.pyplot as plt

    # Test Face
    fig = plt.figure(num=4, clear=True)
    ax = Axes3Ds(fig)
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
    ax = Axes3Ds(fig)
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
    ax = Axes3Ds(fig)
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
    ax = Axes3Ds(fig)
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
    ax = Axes3Ds(fig)
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


    # Draw all figures.
    plt.show()

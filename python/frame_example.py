# this line makes the code compatible with Python 2 and 3
from __future__ import print_function, division

# this line makes Jupyter show figures in the notebook
get_ipython().magic('matplotlib inline')

class FrameError(ValueError):
    """Indicates an error related to Frames."""

class Vector:
    def __init__(self, array, frame=None):
        """A vector is an array of coordinates and a frame of reference.

        array: sequence of coordinates
        frame: Frame object
        """
        self.array = np.asarray(array)
        self.frame = frame

    def __str__(self):
        if self.frame == None:
            return '^{O}%s' % (str(self.array), )
        else:
            return '^{%s}%s' % (str(self.frame), str(self.array))
        
    def __repr__(self):
        return 'Frame(%s, %s)' % (str(self.frame), str(self.array))

    def __add__(self, other):
        if self.frame != other.frame:
            raise FrameError("Vectors must be relative to the same frame.")

        return Vector(self.array + other.array, self.frame)

class Rotation:
    def __init__(self, array):
        self.array = array
    
    def __str__(self):
        return 'Rotation\n%s' % str(self.array)
    
    __repr__ = __str__


    def __neg__(self):
        return Rotation(-self.array)

    def __mul__(self, other):
        """Apply the rotation to a Vector."""
        return np.dot(self.array, other.array)

    __call__ = __mul__

    @staticmethod
    def from_axis(axis, theta):
        x, y, z = np.ravel(axis.array)
        c = np.cos(theta)
        u = 1.0-c
        s = np.sqrt(1.0-c*c)
        xu, yu, zu = x*u, y*u, z*u
        v1 = [x*xu + c, x*yu - z*s, x*zu + y*s]
        v2 = [x*yu + z*s, y*yu + c, y*zu - x*s]
        v3 = [x*zu - y*s, y*zu + x*s, z*zu + c]
        return Rotation(np.array([v1, v2, v3]))

    def to_axis(self):
        # return the equivalent angle-axis as (khat, theta)
        pass

    def transpose(self):
        return Rotation(np.transpose(self.array))

    inverse = transpose
    

class Transform:
    """Represents a transform from one Frame to another."""

    def __init__(self, rot, org, source=None):
        """Instantiates a Transform.

        rot: Rotation object
        org: origin Vector
        source: source Frame
        """
        self.rot = rot
        self.org = org
        self.dest = org.frame
        self.source = source
        self.source.add_transform(self)

    def __str__(self):
        """Returns a string representation of the Transform."""
        if self.dest == None:
            return '%s' % self.source.name
            return '_{%s}^{O}T' % self.source.name
        else:
            return '_{%s}^{%s}T' % (self.source.name, self.dest.name)
        
    __repr__ = __str__
            
    def __mul__(self, other):
        """Applies a Transform to a Vector or Transform."""
        if isinstance(other, Vector):
            return self.mul_vector(other)

        if isinstance(other, Transform):
            return self.mul_transform(other)

    __call__ = __mul__

    def mul_vector(self, p):
        """Applies a Transform to a Vector.

        p: Vector

        Returns: Vector
        """
        if p.frame != self.source:
            raise FrameError(
                "The frame of the vector must be the source of the transform")
        return Vector(self.rot * p, self.dest) + self.org

    def mul_transform(self, other):
        """Applies a Transform to another Transform.

        other: Transform

        Returns Transform
        """
        if other.dest != self.source:
            raise FrameError(
                "This frames source must be the other frame's destination.")

        rot = Rotation(self.rot * other.rot)
        t = Transform(rot, self * other.org, other.source)
        return t

    def inverse(self):
        """Computes the inverse transform.

        Returns: Transform
        """
        irot = self.rot.inverse()
        iorg = Vector(-(irot * self.org), self.source)
        t = Transform(irot, iorg, self.dest)
        return t


class Frame:
    """Represents a frame of reference."""

    # list of Frames
    roster = []
    
    def __init__(self, name):
        """Instantiate a Frame.

        name: string
        """
        self.name = name
        self.transforms = {}
        Frame.roster.append(self)

    def __str__(self): 
        return self.name
    
    __repr__ = __str__

    def add_transform(self, transform):
        """A frames is defined by a Transform relative to another Frame.

        transform: Transform object
        """
        if transform.source != self:
            raise FrameError("Source of the transform must be this Frame.")

        if transform.dest:
            self.transforms[transform.dest] = transform

    def dests(self):
        """Returns a list of the Frames we know how to Transform to."""
        return self.transforms.keys()

origin = Frame('O')
origin

import numpy as np

theta = np.pi/2
xhat = Vector([1, 0, 0], origin)
rx = Rotation.from_axis(xhat, theta)
a = Frame('A')
t_ao = Transform(rx, xhat, a)
t_ao

from IPython.display import Math

def render(obj):
    return Math(str(obj))

render(t_ao)

yhat = Vector([0, 1, 0], a)
ry = Rotation.from_axis(yhat, theta)
b = Frame('B')
t_ba = Transform(ry, yhat, b)
render(t_ba)

zhat = Vector([0, 0, 1], b)
rz = Rotation.from_axis(zhat, theta)
c = Frame('C') 
t_cb = Transform(rz, zhat, c)
render(t_cb)

p_c = Vector([1, 1, 1], c)
render(p_c)

p_b = t_cb(p_c)
render(p_b)

p_a = t_ba(p_b)
render(p_a)

p = t_ao(p_a)
render(p)

import networkx as nx

def add_edges(G, frame):
    for neighbor, transform in frame.transforms.items():
        G.add_edge(frame, neighbor, dict(transform=transform))

def make_graph(frames):
    G = nx.DiGraph()
    for frame in frames:
        add_edges(G, frame)
    return G

frames = Frame.roster
frames

labels = dict([(frame, str(frame)) for frame in frames])
labels

G = make_graph(Frame.roster)
nx.draw(G, labels=labels)

nx.shortest_path(G, c, origin)

cbao = t_ao(t_ba(t_cb))
render(cbao)

p = cbao(p_c)
render(p)

G = make_graph([origin, a, b, c])
nx.draw(G, labels=labels)

nx.shortest_path(G, c, origin)

inv = cbao.inverse()
render(inv)

p_c = inv(p)
render(p_c)


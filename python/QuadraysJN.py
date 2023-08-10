from math import radians, degrees, cos, sin, acos
import math
from operator import add, sub, mul, neg
from collections import namedtuple

XYZ = namedtuple("xyz_vector", "x y z")
IVM = namedtuple("ivm_vector", "a b c d")

root2   = 2.0**0.5

class Qvector:
    """Quadray vector"""

    def __init__(self, arg):
        """Initialize a vector at an (a,b,c,d)"""
        self.coords = self.norm(arg)

    def __repr__(self):
        return repr(self.coords)

    def norm(self, arg):
        """Normalize such that 4-tuple all non-negative members."""
        return IVM(*tuple(map(sub, arg, [min(arg)] * 4))) 
    
    def norm0(self):
        """Normalize such that sum of 4-tuple members = 0"""
        q = self.coords
        return IVM(*tuple(map(sub, q, [sum(q)/4.0] * 4))) 

    @property
    def a(self):
        return self.coords.a

    @property
    def b(self):
        return self.coords.b

    @property
    def c(self):
        return self.coords.c

    @property
    def d(self):
        return self.coords.d
        
    def __mul__(self, scalar):
        """Return vector (self) * scalar."""
        newcoords = [scalar * dim for dim in self.coords]
        return Qvector(newcoords)

    __rmul__ = __mul__ # allow scalar * vector

    def __truediv__(self,scalar):
        """Return vector (self) * 1/scalar"""        
        return self.__mul__(1.0/scalar)
    
    def __add__(self,v1):
        """Add a vector to this vector, return a vector""" 
        newcoords = tuple(map(add, v1.coords, self.coords))
        return Qvector(newcoords)
        
    def __sub__(self,v1):
        """Subtract vector from this vector, return a vector"""
        return self.__add__(-v1)
    
    def __neg__(self):      
        """Return a vector, the negative of this one."""
        return Qvector(tuple(map(neg, self.coords)))
                  
    def dot(self,v1):
        """Return the dot product of self with another vector.
        return a scalar"""
        return 0.5 * sum(map(mul, self.norm0(), v1.norm0()))

    def length(self):
        """Return this vector's length"""
        return self.dot(self) ** 0.5
        
    def xyz(self):
        a,b,c,d     =  self.coords
        k           =  0.5/root2
        xyz         = (k * (a - b - c + d),
                       k * (a - b + c - d),
                       k * (a + b - c - d))
        return Vector(xyz)

class Vector:

    def __init__(self, arg):
        """Initialize a vector at an (x,y,z)"""
        self.xyz = XYZ(*map(float,arg))

    def __repr__(self):
        return repr(self.xyz)
    
    @property
    def x(self):
        return self.xyz.x

    @property
    def y(self):
        return self.xyz.y

    @property
    def z(self):
        return self.xyz.z
        
    def __mul__(self, scalar):
        """Return vector (self) * scalar."""
        newcoords = [scalar * dim for dim in self.xyz]
        return type(self)(newcoords)

    __rmul__ = __mul__ # allow scalar * vector

    def __truediv__(self,scalar):
        """Return vector (self) * 1/scalar"""        
        return self.__mul__(1.0/scalar)
    
    def __add__(self,v1):
        """Add a vector to this vector, return a vector""" 
        newcoords = map(add, v1.xyz, self.xyz)
        return type(self)(newcoords)
        
    def __sub__(self,v1):
        """Subtract vector from this vector, return a vector"""
        return self.__add__(-v1)
    
    def __neg__(self):      
        """Return a vector, the negative of this one."""
        return type(self)(tuple(map(neg, self.xyz)))

    def unit(self):
        return self.__mul__(1.0/self.length())

    def dot(self,v1):
        """Return scalar dot product of this with another vector."""
        return sum(map(mul , v1.xyz, self.xyz))

    def cross(self,v1):
        """Return the vector cross product of this with another vector"""
        newcoords = (self.y * v1.z - self.z * v1.y, 
                     self.z * v1.x - self.x * v1.z,
                     self.x * v1.y - self.y * v1.x )
        return type(self)(newcoords)
    
    def length(self):
        """Return this vector's length"""
        return self.dot(self) ** 0.5

    def quadray(self):
        """return (a, b, c, d) quadray based on current (x, y, z)"""
        x, y, z = self.xyz
        k = 1/root2
        a = k * ((x >= 0)* ( x) + (y >= 0) * ( y) + (z >= 0) * ( z))
        b = k * ((x <  0)* (-x) + (y <  0) * (-y) + (z >= 0) * ( z))
        c = k * ((x <  0)* (-x) + (y >= 0) * ( y) + (z <  0) * (-z))
        d = k * ((x >= 0)* ( x) + (y <  0) * (-y) + (z <  0) * (-z))
        return Qvector((a, b, c, d))

octant0 = Vector((root2/2, root2/2, root2/2))
print(octant0.xyz)
q0 = octant0.quadray()
print(q0)

q0.length()

octant1 = Vector((-root2/2, root2/2, root2/2))  # neighboring octant
diff = octant0 - octant1
diff.length()

q1 = octant1.quadray()
q1

# add up three quadrays and negate their sum, to get the other Qray
a = Qvector((1,0,0,0))
c = Qvector((0,0,1,0))
d = Qvector((0,0,0,1))
v_sum = -(a + c + d)
print("Canonical representation:", v_sum)
print("Alternative expression:  ", v_sum.norm0())


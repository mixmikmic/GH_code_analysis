get_ipython().magic('gui wx')

from numpy import pi, sin, cos, mgrid
from mayavi import mlab

def saddle(t, u):
    x = sin(t)
    y = sin(u)
    z = cos(u) - cos(t)
    return (x, y, z)

dt, du = 0.05*pi, 0.05*pi
t, u = mgrid[-0.5*pi:0.5*pi+dt:dt, -0.5*pi:0.5*pi+du:du]

mlab.mesh(*saddle(t, u))

def ruffle(u, v):
    x = sin(u * cos(v*6)) / cos(v*6)
    y = v + (u * sin(12*v)/2**3)
    z = (1 - cos(u * cos(v*6))) / cos(v*6)
    return (x, y, z)

def wrap(x, y, z):
    x1 = x * cos(y)
    y1 = x * sin(y)
    z1 = z
    return (x1, y1, z1)

def scale(x, y, z, scale_factor=1.0):
    x1 = x * scale_factor
    y1 = y * scale_factor
    z1 = z * scale_factor
    return (x1, y1, z1)

inner = pi/2**5
outer = pi/2

du, dv = 0.05*pi, 0.01*pi
u, v = mgrid[inner:outer+du:du, 0:2*pi+dv:dv]

mlab.clf()
mlab.mesh(*scale(*wrap(*ruffle(u, v)), scale_factor=10.0))

filepath_obj = "/data/notebooks/wrapped_ruffle.obj"
mlab.savefig(filepath_obj)

filepath_stl = "/data/notebooks/wrapped_ruffle.stl"
thickness = 1.0

import subprocess
command = [
    "blender", 
    "--background", 
    "--python", 
    "/data/tools/solidify.py", 
    "--", 
    filepath_obj, 
    filepath_stl,
    str(thickness)
]
subprocess.call(command)




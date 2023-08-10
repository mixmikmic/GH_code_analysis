get_ipython().magic('matplotlib inline')

import subprocess
from matplotlib import pyplot as plt

import pygimli as pg
from pygimli.meshtools import readGmsh

subprocess.call(["wget", "http://www.pygimli.org/_downloads/mesh.geo"])

try:
    subprocess.call(["gmsh", "-2", "-o", "mesh.msh", "mesh.geo"])
    gmsh = True
except OSError:
    print("Gmsh needs to be installed for this example.")
    gmsh = False

if gmsh:
    mesh = readGmsh("mesh.msh", verbose=True)
    pg.show(mesh, mesh.cellMarker(), showLater=True, cmap="BrBG")
    plt.xlim(0,50)
    plt.ylim(-50,0)
else:
    plt.figure()
    plt.title("Gmsh needs to be installed for this example")

plt.show()


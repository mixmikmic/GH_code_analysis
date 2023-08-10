#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#Import packages we need
import numpy as np
from netCDF4 import Dataset
from matplotlib import animation, rc
from matplotlib import pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

from SWESimulators import Common, CDKLM16, SimReader, CTCS, PlotHelper
from SWESimulators import BathymetryAndICs as bic

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

gpu_ctx = Common.CUDAContext()

# open a the netCDF file for reading.
filename = 'data/Nordic-4km_SURF_1h_avg_00.nc?var=h'
ncfile = Dataset(filename,'r') 

for var in ncfile.variables:
    print var

print ("\nAttributes:")    
for attr in ncfile.ncattrs():
    print attr, "\t --> ", ncfile.getncattr(attr)
    
X = ncfile.variables['X']
Y = ncfile.variables['Y']
H = ncfile.variables['h']

# Read netCDF data through THREDDS server
url = 'http://thredds.met.no/thredds/dodsC/fou-hi/nordic4km-1h/Nordic-4km_SURF_1h_avg_00.nc'

ncfile = Dataset(url)

X = ncfile.variables['X']
Y = ncfile.variables['Y']
H = ncfile.variables['h']

print( "shapes: ", X.shape, Y.shape, H.shape)
print( "min/max H: ", np.min(H), np.max(H))
fig = plt.figure(figsize=(6,3))
plt.imshow(H, interpolation="None", origin='lower')
plt.colorbar()

npH = np.array(H)

posH = npH > 15
print type(posH)
fig = plt.figure(figsize=(5, 3))
plt.imshow(posH , interpolation="None", origin='lower')

# Obtaining chunk of ocean between UK and Iceland

atlantic_startX = 0
atlantic_endY = 578

atlantic_startY = 300
atlantic_endX = 300

def plotChunk(H, startX, endX, startY, endY, chunkTitle="Chunk"):
    H_chunk = H[startY:endY, startX:endX]
    print "shape H_chunk:", H_chunk.shape
    
    fig = plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(H, origin='lower')
    plt.title("Global field")
    
    plt.subplot(2,2,3)
    plt.imshow(H > 15, origin='lower')
    plt.title("Global landmask")
    
    plt.subplot(2,2,2)
    plt.imshow(H_chunk, origin='lower')
    plt.title(chunkTitle + " field")
    
    plt.subplot(2,2,4)
    plt.imshow(H_chunk > 15, origin='lower')
    plt.title(chunkTitle + " landmask")
    
    print( "(min, max) of section: ", (np.min(H_chunk), np.max(H_chunk)))
    
plotChunk(npH, atlantic_startX, atlantic_endX, atlantic_startY, atlantic_endY, "Atlantic")
    

# Obtaining chunk of the North Sea

northSea_startX = 350
northSea_endX = 740

northSea_startY = 240
northSea_endY = 420

plotChunk(npH, northSea_startX, northSea_endX, northSea_startY, northSea_endY, "North Sea")
    

# Checking the X and Y variables
#fig = plt.figure(figsize=(3,3))
#plt.plot(X, label='X')
#plt.plot(Y, label='Y')
#plt.legend()

dx = X[1] - X[0]
dy = Y[1] - Y[0]
print("(dx, dy) in the start of the file:  ",(dx, dy))
print("(dx, dy) somewhere else in the file:", (X[400] - X[399], Y[400] - Y[399]))
print "lenght of north sea: ", dx*(northSea_endX - northSea_startX)
print "And that makes roughly sense"

def selectSection(H, startX, endX, startY, endY, chunkTitle="Chunk"):
    H_selection = H[startY:endY, startX:endX]
    nx = endX - startX
    ny = endY - startY
    return H_selection, nx, ny
    
h0, nx, ny = selectSection(npH, atlantic_startX, atlantic_endX, atlantic_startY, atlantic_endY)
#H, nx, ny = selectSection(npH, northSea_startX, northSea_endX, northSea_startY, northSea_endY)

# X and Y are in km, we need m
dx = (X[1] - X[0])*1000
dy = (Y[1] - Y[0])*1000

# Adjusting nx and ny according to boundary condition
nx = nx-20
ny = ny-20

ghostCells = [10,10,10,10]
dataShape = (ny + ghostCells[0] + ghostCells[2], nx + ghostCells[1] + ghostCells[3])
#boundaryConditions = Common.BoundaryConditions(2,2,2,2)
boundaryConditions = Common.BoundaryConditions(3,3,3,3, spongeCells=ghostCells)


dt = 5
g = 9.81
f = 0.00004
r = 0.0
A = 10

eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
u0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32, order='C');
v0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32, order='C'); 

bic.addBump(eta0, nx, ny, dx, dy, 0.2, 0.2, 200, ghostCells)
#bic.addBump(eta0, nx, ny, dx, dy, 0.2, 0.2, 200, ghostCells)
#bic.addBump(eta0, nx, ny, dx, dy, 0.2, 0.2, 200, ghostCells)
#bic.addBump(eta0, nx, ny, dx, dy, 0.2, 0.2, 200, ghostCells)

fig = plt.figure(figsize=(4,4))
plt.imshow(eta0, origin="lower")
plt.colorbar()
plt.title("Initial conditions")
fig = plt.figure(figsize=(4,4))
plt.imshow(h0, origin="lower")
plt.colorbar()
plt.title("Bathymetry")


x_center = dx*nx*0.3
y_center = dy*ny*0.2
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

if 'sim' in globals():
    sim.cleanUp()
reload(CTCS)
reload(PlotHelper)
sim = CTCS.CTCS(gpu_ctx,                 h0, eta0, u0, v0,                 nx, ny,                 dx, dy, dt,                 g, f, r, A,                 boundary_conditions=boundaryConditions )

fig = plt.figure()
eta1, u1, v1 = sim.download(interior_domain_only=True)
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, eta1, u1, v1)

#T = 300
T = 50
def animate(i):
    if (i>0):
        t = sim.step(10.0*dt)
    else:
        t = 0.0
    eta1, u1, v1 = sim.download(interior_domain_only=True)

    plotter.plot(eta1, u1, v1);
    fig.suptitle("CTCS Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print "{:03.0f}".format(100.0*i / T) + " % => t=" + str(t)

anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
plt.close(anim._fig)
anim

ncfile.close()

from __future__ import unicode_literals
#from builtins import str
print (str)
str("heisann", 'utf8')


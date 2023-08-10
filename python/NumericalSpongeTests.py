#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec

import os
import pyopencl
import datetime
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

#Finally, import our simulator
from SWESimulators import FBL, CTCS, KP07, CDKLM16, RecursiveCDKLM16, SimWriter, PlotHelper, Common
from SWESimulators.BathymetryAndICs import *

#Make sure we get compiler output from OpenCL
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

#Set which CL device to use, and disable kernel caching
if (str.lower(sys.platform).startswith("linux")):
    os.environ["PYOPENCL_CTX"] = "0"
else:
    os.environ["PYOPENCL_CTX"] = "1"
os.environ["CUDA_CACHE_DISABLE"] = "1"
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_NO_CACHE"] = "1"

#Create OpenCL context
cl_ctx = pyopencl.create_some_context()
print "Using ", cl_ctx.devices[0].name

#Create output directory for images
imgdir='images_' + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
os.makedirs(imgdir)
print "Saving images to " + imgdir

if 'sim' in globals():
    print "Found sim, so we will clean up!"
    sim.cleanUp()
    print " success :)"
else:
    print "Did not find sim in globals, and that's fine as well!"

reload(Common)
reload(CDKLM16)
    
#Coriolis well balanced reconstruction scheme  - NUMERICAL SPONGE TEST
nx = 10
ny = 20

dx = 200.0
dy = 200.0

dt = 0.95/5.0
g = 9.81

f = 0.00
r = 0.0
wind = Common.WindStressParams(type=99)


#ghosts = np.array([2,2,2,2]) # north, east, south, west
# Numerical Sponge
ghosts = [10, 10, 10, 10]
boundaryConditions = Common.BoundaryConditions(4,4,4,4, spongeCells=ghosts)
validDomain = [None, None, 0, 0]
    

dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

print ("dataShape from notebook: ", dataShape)

waterHeight = 60
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C');

# Bathymetry:
Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')

addCentralBump(h0, nx, ny, dx, dy, ghosts)
for x in range(nx + ghosts[1] + ghosts[3]):
    for y in range(ny + ghosts[0] + ghosts[2]):
        if (x == ghosts[3] or x == nx + ghosts[3]-1) and            (y > ghosts[2] and y < ny + ghosts[2]-1):
            h0[y,x] += 1
        if (y == ghosts[2] or y == ny + ghosts[2]-1) and            (x >= ghosts[3] and x <= nx + ghosts[3]-1):
            h0[y,x] += 1


fig = plt.figure(figsize=(3,3))
#plt.imshow(h0[ghosts[2]:-ghosts[0], ghosts[3]:-ghosts[1]], interpolation="None")
plt.imshow(h0, interpolation="None")
plt.colorbar()
plt.title("Initial contidions before applying numerical sponge")

#Initialize simulator
reload(CDKLM16)
reload(KP07)
reload(Common)
#sim = KP07.KP07(cl_ctx, \
sim = CDKLM16.CDKLM16(cl_ctx,                 h0, u0, v0,                 Bi,                 nx, ny,                 dx, dy, dt,                 g, f, r,                 wind_stress=wind,                 boundary_conditions=boundaryConditions)

t = sim.step(0.0)
h1, u1, v1 = sim.download()
fig = plt.figure(figsize=(3,3))
plt.imshow(h1, interpolation="None")
plt.colorbar()
plt.title("Initial conditions after applying numerical sponge")

if 'sim' in globals():
    sim.cleanUp()

reload(Common)
reload(CDKLM16)
    
#Coriolis well balanced reconstruction scheme  - NUMERICAL SPONGE TEST
nx = 10
ny = 20

dx = 200.0
dy = 200.0

dt = 0.95/5.0
g = 9.81

f = 0.00
r = 0.0
wind = Common.WindStressParams(type=99)


#ghosts = np.array([2,2,2,2]) # north, east, south, west
fig = plt.figure()
fig.title = "Mixing boundary conditions"

for i in range(7):
    if i == 0:
        msg = "Sponge in north"
        ghosts = [10, 2, 2, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(4,2,1,2, spongeCells=ghosts)
    elif i == 1:
        msg = "Sponge in east"
        ghosts = [2, 10, 2, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(2,4,2,1, spongeCells=ghosts)
    elif i == 2:
        msg = "Sponge in south"
        ghosts = [2, 2, 10, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,2,4,2, spongeCells=ghosts)
    elif i == 3:
        msg = "Sponge in west"
        ghosts = [2, 2, 2, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(2,1,2,4, spongeCells=ghosts)
    elif i == 4:
        msg = "Sponge in west and north"
        ghosts = [10, 2, 2, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(4,1,1,4, spongeCells=ghosts)
    elif i == 5:
        msg = "Sponge in east and south"
        ghosts = [2, 10, 10, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,4,4,1, spongeCells=ghosts)
    elif i == 6:
        msg = "Sponge in east and south and west"
        ghosts = [2, 10, 10, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,4,4,4, spongeCells=ghosts)
    
    print "---------------\n" + msg
    validDomain = [None, None, 0, 0] # Plot all cells (including ghost cells)
    

    dataShape = (ny + ghosts[0]+ghosts[2], 
                 nx + ghosts[1]+ghosts[3])


    waterHeight = 60
    h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
    u0 = np.zeros(dataShape, dtype=np.float32, order='C');
    v0 = np.zeros(dataShape, dtype=np.float32, order='C');

    # Bathymetry:
    Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')

    addCentralBump(h0, nx, ny, dx, dy, ghosts)
    for x in range(nx + ghosts[1] + ghosts[3]):
        for y in range(ny + ghosts[0] + ghosts[2]):
            if (x == ghosts[3] or x == nx + ghosts[3]-1) and                (y > ghosts[2] and y < ny + ghosts[2]-1):
                h0[y,x] += 1
            if (y == ghosts[2] or y == ny + ghosts[2]-1) and                (x >= ghosts[3] and x <= nx + ghosts[3]-1):
                h0[y,x] += 1
    #print "From notebook, shape of h: ", h0.shape


    plt.subplot(7,2,i*2+1)
    #plt.imshow(h0[ghosts[2]:-ghosts[0], ghosts[3]:-ghosts[1]], interpolation="None")
    plt.imshow(h0, interpolation="None")
    plt.colorbar()
    plt.title(msg)

    #Initialize simulator
    reload(CDKLM16)
    reload(KP07)
    reload(Common)
    #sim = KP07.KP07(cl_ctx, \
    sim = CDKLM16.CDKLM16(cl_ctx,                     h0, u0, v0,                     Bi,                     nx, ny,                     dx, dy, dt,                     g, f, r,                     wind_stress=wind,                     boundary_conditions=boundaryConditions)

    t = sim.step(0.0)
    h1, u1, v1 = sim.download()
    sim.cleanUp()
    plt.subplot(7,2,i*2+2)
    plt.imshow(h1, interpolation="None")
    plt.colorbar()
    plt.title(msg)

if 'sim' in globals():
    sim.cleanUp()
    
#Coriolis well balanced reconstruction scheme
nx = 30
ny = 60

dx = 200.0
dy = 200.0

dt = 0.95/5.0
g = 9.81

f = 0.00
r = 0.0
wind = Common.WindStressParams(type=99)

# Numerical Sponge
ghosts = [10, 10, 10, 10] # north, east, south, west
boundaryConditions = Common.BoundaryConditions(4,4,4,4, spongeCells=ghosts)
validDomain = [None, None, 0, 0]
    
dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

waterHeight = 60
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
addLowerLeftBump(h0, nx, ny, dx, dy, ghosts)

u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C');

# Bathymetry:
Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')



#Initialize simulator
reload(CDKLM16)
reload(Common)
sim = CDKLM16.CDKLM16(cl_ctx,                 h0, u0, v0, Bi,nx, ny, dx, dy, dt, g, f, r,                 wind_stress=wind, boundary_conditions=boundaryConditions,                 write_netcdf=False)

#Calculate radius from center of bump for plotting
x_center = dx*(nx+ghosts[0]+ghosts[2])/2.0
y_center = dy*(ny+ghosts[1]+ghosts[3])/2.0
y_coords, x_coords = np.mgrid[0:(ny+ghosts[0]+ghosts[2])*dy:dy, 0:(nx+ghosts[1]+ghosts[3])*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                h0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - waterHeight, 
                                u0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                                v0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]])

T = 250

def animate(i):
    if (i>0):
        #t = sim.step(10.0)
        t = sim.step(5.0)
    else:
        t = 0.0
    h1, u1, v1 = sim.download()   
    brighten = 10 # Increase the values in the animation
    
    plotter.plot(brighten*(h1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - waterHeight), 
                 brighten*u1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                 brighten*v1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]]);
    fig.suptitle("CDKLM16 Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print "{:03.0f}".format(100*i / T) + " % => t=" + str(t) + "\t(Min, max) h: " + str((np.min(h1),np.max(h1))) +         "\tMax (u, v): " + str((np.max(u1), np.max(v1)))
        fig.savefig(imgdir + "/{:010.0f}_cdklm16.png".format(t))
             
anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
plt.close(anim._fig)
anim

#h2, u2, v2 = sim.download()
print "Initial water volume:      ", sum(sum(h0))
print "Initial bump volume:       ", sum(sum(h0 - waterHeight))
print "steady-state water volume: ", sum(sum(h2))
print "Water loss:                ", sum(sum(h0)) - sum(sum(h0 - waterHeight)) - sum(sum(h2))
sim.cleanUp()

if 'sim' in globals():
    sim.cleanUp()
    
    
print "Using (ny, nx): ", (ny, nx)
print "with ghosts: ", ghosts
#Coriolis well balanced reconstruction scheme
waterHeight = 60
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
addCentralBump(h0, nx, ny, dx, dy, ghosts)

f = 0.01

#Initialize simulator
reload(CDKLM16)
reload(Common)
sim = CDKLM16.CDKLM16(cl_ctx,                 h0, u0, v0, Bi,nx, ny, dx, dy, dt, g, f, r,                 wind_stress=wind, boundary_conditions=boundaryConditions,                 write_netcdf=True)

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                h0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - waterHeight, 
                                u0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                                v0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]])

T = 250

def animate(i):
    if (i>0):
        #t = sim.step(10.0)
        t = sim.step(5.0)
    else:
        t = 0.0
    h1, u1, v1 = sim.download()   
    brighten = 10 # Increase the values in the animation
    
    plotter.plot(brighten*(h1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - waterHeight), 
                 brighten*u1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                 brighten*v1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]]);
    fig.suptitle("CDKLM16 Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%30 == 0):
        print "{:03.0f}".format(100*i / T) + " % => t=" + str(t) + "\t(Min, max) h: " + str((np.min(h1),np.max(h1))) +         "\tMax (u, v): " + str((np.max(u1), np.max(v1)))
        fig.savefig(imgdir + "/{:010.0f}_cdklm16.png".format(t))
             
anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
plt.close(anim._fig)
anim

#h2, u2, v2 = sim.download()
print "Initial water volume:      ", sum(sum(h0))
print "Initial bump volume:       ", sum(sum(h0 - waterHeight))
print "steady-state water volume: ", sum(sum(h2))
print "Water loss:                ", sum(sum(h0)) - sum(sum(h0 - waterHeight)) - sum(sum(h2))
sim.cleanUp()

if 'sim' in globals():
    sim.cleanUp()

reload(Common)
reload(KP07)
    
#Coriolis well balanced reconstruction scheme  - NUMERICAL SPONGE TEST
nx = 10
ny = 20

dx = 200.0
dy = 200.0

dt = 0.95/5.0
g = 9.81

f = 0.00
r = 0.0
wind = Common.WindStressParams(type=99)


#ghosts = np.array([2,2,2,2]) # north, east, south, west
fig = plt.figure()
fig.title = "Mixing boundary conditions"

for i in range(8):
    if i == 0:
        msg = "Sponge in all over"
        ghosts = [10, 10, 10, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(4,4,4,4, spongeCells=ghosts)
    elif i == 1:
        msg = "Sponge in east"
        ghosts = [2, 10, 2, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(2,4,2,1, spongeCells=ghosts)
    elif i == 2:
        msg = "Sponge in south"
        ghosts = [2, 2, 10, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,2,4,2, spongeCells=ghosts)
    elif i == 3:
        msg = "Sponge in west"
        ghosts = [2, 2, 2, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(2,1,2,4, spongeCells=ghosts)
    elif i == 4:
        msg = "Sponge in north"
        ghosts = [10, 2, 2, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(4,2,1,2, spongeCells=ghosts)
    elif i == 5:
        msg = "Sponge in east and south"
        ghosts = [2, 10, 10, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,4,4,1, spongeCells=ghosts)
    elif i == 6:
        msg = "Sponge in west and north"
        ghosts = [10, 2, 2, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(4,1,1,4, spongeCells=ghosts)
    elif i == 7:
        msg = "Sponge in east and south and west"
        ghosts = [2, 10, 10, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,4,4,4, spongeCells=ghosts)
    
    print "---------------\n" + msg
    
    validDomain = [None, None, 0, 0] # Plot all cells (including ghost cells)
    

    dataShape = (ny + ghosts[0]+ghosts[2], 
                 nx + ghosts[1]+ghosts[3])


    waterHeight = 60
    h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
    u0 = np.zeros(dataShape, dtype=np.float32, order='C');
    v0 = np.zeros(dataShape, dtype=np.float32, order='C');

    # Bathymetry:
    Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')

    addCentralBump(h0, nx, ny, dx, dy, ghosts)
    for x in range(nx + ghosts[1] + ghosts[3]):
        for y in range(ny + ghosts[0] + ghosts[2]):
            if (x == ghosts[3] or x == nx + ghosts[3]-1) and                (y > ghosts[2] and y < ny + ghosts[2]-1):
                h0[y,x] += 1
            if (y == ghosts[2] or y == ny + ghosts[2]-1) and                (x >= ghosts[3] and x <= nx + ghosts[3]-1):
                h0[y,x] += 1
    #print "From notebook, shape of h: ", h0.shape


    plt.subplot(8,2,i*2+1)
    #plt.imshow(h0[ghosts[2]:-ghosts[0], ghosts[3]:-ghosts[1]], interpolation="None")
    plt.imshow(h0, interpolation="None")
    plt.colorbar()
    plt.title(msg)

    #Initialize simulator
    reload(KP07)
    reload(Common)
    sim = KP07.KP07(cl_ctx,                     h0, Bi, u0, v0,                     nx, ny,                     dx, dy, dt,                     g, f, r,                     wind_stress=wind,                     boundary_conditions=boundaryConditions)

    t = sim.step(0.0)
    h1, u1, v1 = sim.download()
    sim.cleanUp()
    plt.subplot(8,2,i*2+2)
    plt.imshow(h1, interpolation="None")
    plt.colorbar()
    plt.title(msg)

if 'sim' in globals():
    sim.cleanUp()
    
#Coriolis well balanced reconstruction scheme
nx = 30
ny = 60

dx = 200.0
dy = 200.0

dt = 0.95/5.0
g = 9.81

f = 0.00
r = 0.0
wind = Common.WindStressParams(type=99)

# Numerical Sponge
ghosts = [10, 10, 10, 10] # north, east, south, west
boundaryConditions = Common.BoundaryConditions(4,4,4,4, spongeCells=ghosts)
validDomain = [None, None, 0, 0]
    
dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

waterHeight = 60
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
addLowerLeftBump(h0, nx, ny, dx, dy, ghosts)

u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C');

# Bathymetry:
Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')



#Initialize simulator
reload(KP07)
reload(Common)
sim = KP07.KP07(cl_ctx,                 h0, Bi, u0, v0,nx, ny, dx, dy, dt, g, f, r,                 wind_stress=wind, boundary_conditions=boundaryConditions,                 write_netcdf=False)

#Calculate radius from center of bump for plotting
x_center = dx*(nx+ghosts[0]+ghosts[2])/2.0
y_center = dy*(ny+ghosts[1]+ghosts[3])/2.0
y_coords, x_coords = np.mgrid[0:(ny+ghosts[0]+ghosts[2])*dy:dy, 0:(nx+ghosts[1]+ghosts[3])*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                h0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - waterHeight, 
                                u0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                                v0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]])

T = 250

def animate(i):
    if (i>0):
        #t = sim.step(10.0)
        t = sim.step(5.0)
    else:
        t = 0.0
    h1, u1, v1 = sim.download()   
    brighten = 10 # Increase the values in the animation
    
    plotter.plot(brighten*(h1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - waterHeight), 
                 brighten*u1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                 brighten*v1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]]);
    fig.suptitle("KP07 Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print "{:03.0f}".format(100*i / T) + " % => t=" + str(t) + "\t(Min, max) h: " + str((np.min(h1),np.max(h1))) +         "\tMax (u, v): " + str((np.max(u1), np.max(v1)))
        fig.savefig(imgdir + "/{:010.0f}_kp07.png".format(t))
             
anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
plt.close(anim._fig)
anim

h2, u2, v2 = sim.download()
print "Initial water volume:      ", sum(sum(h0))
print "Initial bump volume:       ", sum(sum(h0 - waterHeight))
print "steady-state water volume: ", sum(sum(h2))
print "Water loss:                ", sum(sum(h0)) - sum(sum(h0 - waterHeight)) - sum(sum(h2))
sim.cleanUp()

if 'sim' in globals():
    sim.cleanUp()
    
    
print "Using (ny, nx): ", (ny, nx)
print "with ghosts: ", ghosts
#Coriolis well balanced reconstruction scheme
waterHeight = 60
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
addCentralBump(h0, nx, ny, dx, dy, ghosts)

f = 0.01

#Initialize simulator
reload(KP07)
reload(Common)
sim = KP07.KP07(cl_ctx,                 h0, Bi, u0, v0, nx, ny, dx, dy, dt, g, f, r,                 wind_stress=wind, boundary_conditions=boundaryConditions,                 write_netcdf=True)

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                h0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - waterHeight, 
                                u0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                                v0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]])

T = 250

def animate(i):
    if (i>0):
        #t = sim.step(10.0)
        t = sim.step(5.0)
    else:
        t = 0.0
    h1, u1, v1 = sim.download()   
    brighten = 10 # Increase the values in the animation
    
    plotter.plot(brighten*(h1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - waterHeight), 
                 brighten*u1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                 brighten*v1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]]);
    fig.suptitle("CDKLM16 Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%30 == 0):
        print "{:03.0f}".format(100*i / T) + " % => t=" + str(t) + "\t(Min, max) h: " + str((np.min(h1),np.max(h1))) +         "\tMax (u, v): " + str((np.max(u1), np.max(v1)))
        fig.savefig(imgdir + "/{:010.0f}_cdklm16.png".format(t))
             
anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
plt.close(anim._fig)
anim

sim.cleanUp()

if 'sim' in globals():
    sim.cleanUp()

reload(Common)
reload(CTCS)
    
#Centered in time, centered in space scheme  - NUMERICAL SPONGE TEST
nx = 10
ny = 20

dx = 200.0
dy = 200.0

dt = 0.95/5.0
g = 9.81

f = 0.00
r = 0.0
A = 1.0
wind = Common.WindStressParams(type=99)


#ghosts = np.array([2,2,2,2]) # north, east, south, west
fig = plt.figure()
fig.title = "Mixing boundary conditions"

for i in range(8):
    if i == 0:
        msg = "Sponge in all over"
        ghosts = [10, 10, 10, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(4,4,4,4, spongeCells=ghosts)
    if i == 1:
        msg = "Sponge in north"
        ghosts = [10, 2, 2, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(4,2,1,2, spongeCells=ghosts)
    elif i == 2:
        msg = "Sponge in east"
        ghosts = [2, 10, 2, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(2,4,2,1, spongeCells=ghosts)
    elif i == 3:
        msg = "Sponge in south"
        ghosts = [2, 2, 10, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,2,4,2, spongeCells=ghosts)
    elif i == 4:
        msg = "Sponge in west"
        ghosts = [2, 2, 2, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(2,1,2,4, spongeCells=ghosts)
    elif i == 5:
        msg = "Sponge in west and north"
        ghosts = [10, 2, 2, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(4,1,1,4, spongeCells=ghosts)
    elif i == 6:
        msg = "Sponge in east and south"
        ghosts = [2, 10, 10, 2] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,4,4,1, spongeCells=ghosts)
    elif i == 7:
        msg = "Sponge in east and south and west"
        ghosts = [2, 10, 10, 10] # north, east, south, west
        boundaryConditions = Common.BoundaryConditions(1,4,4,4, spongeCells=ghosts)
    
    print "---------------\n" + msg
    validDomain = [None, None, 0, 0] # Plot all cells (including ghost cells)
    

    dataShape = (ny + ghosts[0]+ghosts[2], 
                 nx + ghosts[1]+ghosts[3])


    waterHeight = 60
    h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
    eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
    u0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32, order='C');
    v0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32, order='C');
    
    addCentralBump(eta0, nx, ny, dx, dy, ghosts)
    for x in range(nx + ghosts[1] + ghosts[3]):
        for y in range(ny + ghosts[0] + ghosts[2]):
            if (x == ghosts[3] or x == nx + ghosts[3]-1) and                (y > ghosts[2] and y < ny + ghosts[2]-1):
                eta0[y,x] += 1
            if (y == ghosts[2] or y == ny + ghosts[2]-1) and                (x >= ghosts[3] and x <= nx + ghosts[3]-1):
                eta0[y,x] += 1
    for x in range(nx+1 + ghosts[1] + ghosts[3]):
        for y in range(ny + ghosts[0] + ghosts[2]):
            if (x == ghosts[3] or x == nx+1 + ghosts[3]-1) and                (y > ghosts[2] and y < ny + ghosts[2]-1):
                u0[y,x] += 0.1*y*x
            if (y == ghosts[2] or y == ny + ghosts[2]-1) and                (x >= ghosts[3] and x <= nx+1 + ghosts[3]-1):
                u0[y,x] += 0.1*x*y
    for x in range(nx + ghosts[1] + ghosts[3]):
        for y in range(ny+1 + ghosts[0] + ghosts[2]):
            if (x == ghosts[3] or x == nx + ghosts[3]-1) and                (y > ghosts[2] and y < ny+1 + ghosts[2]-1):
                v0[y,x] -= 1
            if (y == ghosts[2] or y == ny+1 + ghosts[2]-1) and                (x >= ghosts[3] and x <= nx + ghosts[3]-1):
                v0[y,x] -= 1
    #print "From notebook, shape of h: ", h0.shape


    plt.subplot(8,2,i*2+1)
    #plt.imshow(h0[ghosts[2]:-ghosts[0], ghosts[3]:-ghosts[1]], interpolation="None")
    plt.imshow(u0, interpolation="None")
    plt.colorbar()
    plt.title(msg)

    #Initialize simulator
    reload(CTCS)
    sim = CTCS.CTCS(cl_ctx,                    h0, eta0, u0, v0,                    nx, ny,                    dx, dy, dt,                    g, f, r, A,                    wind_stress=wind,                    boundary_conditions=boundaryConditions)

    t = sim.step(0.00001)
    eta1, u1, v1 = sim.download()
    sim.cleanUp()    
    plt.subplot(8,2,i*2+2)
    plt.imshow(u1, interpolation="None")
    plt.colorbar()
    plt.title(msg)

if 'sim' in globals():
    sim.cleanUp()
    
#Coriolis well balanced reconstruction scheme
nx = 30
ny = 60

dx = 200.0
dy = 200.0

dt = 0.95/5.0
g = 9.81
A = 1.0

f = 0.00
r = 0.0
wind = Common.WindStressParams(type=99)

# Numerical Sponge
ghosts = [10, 10, 10, 10] # north, east, south, west
boundaryConditions = Common.BoundaryConditions(4,4,4,4, spongeCells=ghosts)
validDomain = [None, None, 0, 0]
    
dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

waterHeight = 60
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
u0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32, order='C');
v0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32, order='C');
addLowerLeftBump(eta0, nx, ny, dx, dy, ghosts)


#Initialize simulator
reload(CTCS)
reload(Common)
sim = CTCS.CTCS(cl_ctx,                    h0, eta0, u0, v0,                    nx, ny, dx, dy, dt,                    g, f, r, A,                    wind_stress=wind,                    boundary_conditions=boundaryConditions,
                   write_netcdf=True)

#Calculate radius from center of bump for plotting
x_center = dx*(nx+ghosts[0]+ghosts[2])/2.0
y_center = dy*(ny+ghosts[1]+ghosts[3])/2.0
y_coords, x_coords = np.mgrid[0:(ny+ghosts[0]+ghosts[2])*dy:dy, 0:(nx+ghosts[1]+ghosts[3])*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] , 
                                u0[validDomain[2]:validDomain[0], validDomain[3]+1:validDomain[1]], 
                                v0[validDomain[2]+1:validDomain[0], validDomain[3]:validDomain[1]],
                                interpolation_type="None")

T = 100
def animate(i):
    if (i>0):
        #t = sim.step(10.0)
        t = sim.step(5.0)
    else:
        t = 0.0
    eta1, u1, v1 = sim.download()   
    brighten = 1 # Increase the values in the animation
    
    eta1[:, ghosts[3]-1] = 61
    eta1[:, nx + ghosts[3] ] = 61
    eta1[ghosts[2]-1, :] = 61
    eta1[ny + ghosts[2], :] = 61
    
    u1[:, ghosts[3]-1] = 61
    u1[:, nx + ghosts[3]+1 ] = 61
    u1[ghosts[2]-1, :] = 61
    u1[ny + ghosts[2], :] = 61
    
    v1[:, ghosts[3]-1] = 61
    v1[:, nx + ghosts[3] ] = 61
    v1[ghosts[2]-1, :] = 61
    v1[ny + ghosts[2]+1, :] = 61
    
    plotter.plot(brighten*(eta1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]]), 
                 brighten*u1[validDomain[2]:validDomain[0], validDomain[3]+1:validDomain[1]], 
                 brighten*v1[validDomain[2]+1:validDomain[0], validDomain[3]:validDomain[1]]);
    fig.suptitle("CTCS Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print "{:03.0f}".format(100*i / T) + " % => t=" + str(t) + "\t(Min, max) h: " + str((np.min(eta1),np.max(eta1))) +         "\tMax (u, v): " + str((np.max(u1), np.max(v1)))
        fig.savefig(imgdir + "/{:010.0f}_ctcs.png".format(t))
             
anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
plt.close(anim._fig)
anim

sim.cleanUp()

x = np.linspace(1, 10, 10)
plt.plot(x, 1 - np.tanh((x-1.0)/2.7), '*')
for i in x:
    print (i, 1 - np.tanh((i-1)/2.0), 1 - np.tanh((i-1)/2.7), 1 - np.tanh((i-1)/3.))


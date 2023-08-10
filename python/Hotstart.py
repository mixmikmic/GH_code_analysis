#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
#%config InlineBackend.figure_format = 'retina'

#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt

import os
import pycuda.driver as cuda
import datetime
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

#Set large figure sizes
#rc('figure', figsize=(16.0, 12.0))
#rc('animation', html='html5')
plt.rcParams["animation.html"] = "jshtml"

#Import our simulator
from SWESimulators import FBL, CTCS, KP07, CDKLM16, SimWriter, SimReader, PlotHelper, Common, WindStress, IPythonMagic
#Import initial condition and bathymetry generating functions:
from SWESimulators.BathymetryAndICs import *

get_ipython().run_line_magic('setup_logging', '--out hotstart.log')
get_ipython().run_line_magic('cuda_context_handler', 'gpu_ctx')

#Create output directory for images
imgdir='images_' + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
os.makedirs(imgdir)
print("Saving images to " + imgdir)

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()
    
# Forward backward linear
nx = 100
ny = 200

dx = 200.0
dy = 200.0

width = nx * dx
height = ny * dy

dt = 1
g = 9.81
r = 0.0
f = 0.01

wind = WindStress.WindStress()

bcSettings = 1
if (bcSettings == 1):
    boundaryConditions = Common.BoundaryConditions()
    ghosts = [0,0,0,0] # north, east, south, west
    validDomain = [None, None, 0, 0]
elif (bcSettings == 2):
    boundaryConditions = Common.BoundaryConditions(2,2,2,2)
    ghosts = [1,1,0,0] # Both periodic
    validDomain = [-1, -1, 0, 0]
elif bcSettings == 3:
    boundaryConditions = Common.BoundaryConditions(2,1,2,1)
    ghosts = [1,0,0,0] # periodic north-south
    validDomain = [-1, None, 0, 0]
else:
    boundaryConditions = Common.BoundaryConditions(1,2,1,2)
    ghosts = [0,1,0,0] # periodic east-west
    validDomain = [None, -1, 0, 0]


h0 = np.ones((ny+ghosts[0], nx+ghosts[1]), dtype=np.float32) * 60;
#addTopographyBump(h0, nx, ny, dx, dy, ghosts, 40)

eta0 = np.zeros((ny+ghosts[0], nx+ghosts[1]), dtype=np.float32);
u0 = np.zeros((ny+ghosts[0], nx+1), dtype=np.float32);
v0 = np.zeros((ny+1, nx+ghosts[1]), dtype=np.float32);

#Create bump in to lower left of domain for testing
addCentralBump(eta0, nx, ny, dx, dy, ghosts)
    

#Initialize simulator
sim = FBL.FBL(gpu_ctx,                 h0, eta0, u0, v0,                 nx, ny,                 dx, dy, dt,                 g, f, r,                 wind_stress=wind,                 boundary_conditions=boundaryConditions,                 write_netcdf=True)


#Calculate radius from center of bump for plotting
x_center = dx*nx*0.3
y_center = dy*ny*0.2
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))
   
fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius,                                 eta0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]],                                u0[validDomain[2]:validDomain[0], :],                                 v0[:, validDomain[3]:validDomain[1]])

T = 50
def animate(i):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    eta1, u1, v1 = sim.download()

    plotter.plot(eta1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]],                 u1[validDomain[2]:validDomain[0], :],                  v1[:, validDomain[3]:validDomain[1]]);
    fig.suptitle("FBL Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print("{:03.0f}".format(100.0*i / T) + " % => t=" + str(t))
        fig.savefig(imgdir + "/{:010.0f}_fbl.png".format(t))

#anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
#plt.close(anim._fig)
#anim

#eta1, u1, v1 = sim.download()
    
for i in range(10):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    eta1, u1, v1 = sim.download()

    print("i: " + str(i) + " t: " + str(t))

# Close nc-file
sim.cleanUp()

# utility function to print last produced nc-file
nc_files = glob.glob(os.path.join(".", "netcdf_" + datetime.datetime.now().strftime("%Y_%m_%d"), "*"))
if not nc_files:
    raise Exception("No nc-files produced today!")
fbl_nc_files = filter(lambda k: 'FBL' in k, nc_files)
print(fbl_nc_files)
print(max(fbl_nc_files))

nc_files = glob.glob(os.path.join(".", "netcdf_" + datetime.datetime.now().strftime("%Y_%m_%d"), "*"))
fbl_nc_files = filter(lambda k: 'FBL' in k, nc_files)
if not fbl_nc_files:
    raise Exception("No nc-files produced today!")
filename =  max(fbl_nc_files)

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()

#Initialize simulator at time time0, using last recorded time step as initial conditions
sim = FBL.FBL.fromfilename(gpu_ctx,                 filename,                 cont_write_netcdf=True)

nx = sim.nx
ny = sim.ny

dx = sim.dx
dy = sim.dy

eta0, u0, v0 = sim.download()

#Calculate radius from center of bump for plotting
x_center = dx*nx*0.3
y_center = dy*ny*0.2
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))
   
fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius,                                 eta0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]],                                u0[validDomain[2]:validDomain[0], :],                                 v0[:, validDomain[3]:validDomain[1]])

T = 50
def animate(i):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    eta1, u1, v1 = sim.download()

    plotter.plot(eta1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]],                 u1[validDomain[2]:validDomain[0], :],                  v1[:, validDomain[3]:validDomain[1]]);
    fig.suptitle("FBL Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print("{:03.0f}".format(100.0*i / T) + " % => t=" + str(t))
        fig.savefig(imgdir + "/{:010.0f}_fbl.png".format(t))

#anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
#plt.close(anim._fig)
#anim

#eta1, u1, v1 = sim.download()
    
for i in range(10):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = sim.t
    eta1, u1, v1 = sim.download()

    print("i: " + str(i) + " t: " + str(t))
    
fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius,                                eta1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]],                                u1[validDomain[2]:validDomain[0], :],                                 v1[:, validDomain[3]:validDomain[1]])

# Close nc-file
sim.cleanUp()

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()
    
#Centered in time, centered in space
nx = 100
ny = 200

dx = 200.0
dy = 200.0

width = nx * dx
height = ny * dy

dt = 1
g = 9.81
r = 0.0
A = 1
f = 0.01

wind = WindStress.WindStress()

bcSettings = 1
ghosts = [1,1,1,1] # north, east, south, west
if (bcSettings == 1):
    boundaryConditions = Common.BoundaryConditions()
    # Wall boundary conditions
elif (bcSettings == 2):
    # periodic boundary conditions
    boundaryConditions = Common.BoundaryConditions(2,2,2,2)
elif bcSettings == 3:
    # periodic north-south
    boundaryConditions = Common.BoundaryConditions(2,1,2,1)
else:
    # periodic east-west
    boundaryConditions = Common.BoundaryConditions(1,2,1,2)

h0 = np.ones((ny+2, nx+2), dtype=np.float32, order='C') * 60;
#addTopographyBump(h0, nx, ny, dx, dy, ghosts, 40)

eta0 = np.zeros((ny+2, nx+2), dtype=np.float32, order='C');
u0 = np.zeros((ny+2, nx+1+2), dtype=np.float32, order='C');
v0 = np.zeros((ny+1+2, nx+2), dtype=np.float32, order='C');

#Create bump in to lower left of domain for testing
x_center = dx*nx*0.3
y_center = dy*ny*0.2
makeCentralBump(eta0, 0.0, nx, ny, dx, dy, ghosts)
#makeUpperCornerBump(eta0, nx, ny, dx, dy, ghosts)
#addDualVortexStaggered(eta0, u0, v0, nx, ny, dx, dy, ghosts)
    

#Initialize simulator
sim = CTCS.CTCS(gpu_ctx,                 h0, eta0, u0, v0,                 nx, ny,                 dx, dy, dt,                 g, f, r, A,                 wind_stress=wind,                 boundary_conditions=boundaryConditions,                 write_netcdf=True)


#Calculate radius from center of bump for plotting
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

ghosts = [-1,1,-1,1]

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, eta0[1:-1, 1:-1], u0[1:-1, :], v0[:, 1:-1])

T = 50
def animate(i):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    eta1, u1, v1 = sim.download()

    plotter.plot(eta1[1:-1, 1:-1], u1[1:-1, :], v1[:, 1:-1]);
    fig.suptitle("CTCS Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print("{:03.0f}".format(100.0*i / T) + " % => t=" + str(t))
        fig.savefig(imgdir + "/{:010.0f}_ctcs.png".format(t))

#anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
#plt.close(anim._fig)
#anim

#eta1, u1, v1 = sim.download()
    
for i in range(10):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    eta1, u1, v1 = sim.download()

    print("i: " + str(i) + " t: " + str(t))

# Close nc-file
sim.cleanUp()

# utility function to print last produced nc-file
nc_files = glob.glob(os.path.join(".", "netcdf_" + datetime.datetime.now().strftime("%Y_%m_%d"), "*"))
if not nc_files:
    raise Exception("No nc-files produced today!")
ctcs_nc_files = filter(lambda k: 'CTCS' in k, nc_files)
print(ctcs_nc_files)
print(max(ctcs_nc_files))

nc_files = glob.glob(os.path.join(".", "netcdf_" + datetime.datetime.now().strftime("%Y_%m_%d"), "*"))
ctcs_nc_files = filter(lambda k: 'CTCS' in k, nc_files)
if not ctcs_nc_files:
    raise Exception("No nc-files produced today!")
filename =  max(ctcs_nc_files)

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()

#Initialize simulator at time time0, using last recorded time step as initial conditions
sim = CTCS.CTCS.fromfilename(gpu_ctx,                 filename,                 cont_write_netcdf=True)

nx = sim.nx
ny = sim.ny

dx = sim.dx
dy = sim.dy

eta0, u0, v0 = sim.download()

#Calculate radius from center of bump for plotting
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_center = dx*nx*0.3
y_center = dy*ny*0.2
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

ghosts = [-1,1,-1,1]

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, eta0[1:-1, 1:-1], u0[1:-1, :], v0[:, 1:-1])

T = 50
def animate(i):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    eta1, u1, v1 = sim.download()

    plotter.plot(eta1[1:-1, 1:-1], u1[1:-1, :], v1[:, 1:-1]);
    fig.suptitle("CTCS Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print("{:03.0f}".format(100.0*i / T) + " % => t=" + str(t))
        fig.savefig(imgdir + "/{:010.0f}_ctcs.png".format(t))

#anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
#plt.close(anim._fig)
#anim

#eta1, u1, v1 = sim.download()
    
for i in range(10):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = sim.t
    eta1, u1, v1 = sim.download()

    print("i: " + str(i) + " t: " + str(t))
    
fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius,                                eta1[1:-1, 1:-1], u1[1:-1, :], v1[:, 1:-1])

# Close nc-file
sim.cleanUp()

# DEFINE PARAMETERS

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()
    
# Kurganov-Petrova 2007
nx = 100
ny = 200

dx = 200.0
dy = 200.0

width = nx * dx
height = ny * dy

dt = 1
g = 9.81
r = 0.0
f = 0.01

wind = WindStress.WindStress()

bcSettings = 1
ghosts = np.array([2,2,2,2]) # north, east, south, west
validDomain = np.array([2,2,2,2])
if (bcSettings == 1):
    boundaryConditions = Common.BoundaryConditions()
elif (bcSettings == 2):
    boundaryConditions = Common.BoundaryConditions(2,2,2,2)
elif bcSettings == 3:
    # Periodic NS
    boundaryConditions = Common.BoundaryConditions(2,1,2,1)
else:
    # Periodic EW
    boundaryConditions = Common.BoundaryConditions(1,2,1,2)
    
dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])
waterHeight = 60
eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C');

Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * waterHeight;

addCentralBump(eta0, nx, ny, dx, dy, validDomain)

   

#Initialize simulator
sim = KP07.KP07(gpu_ctx,                 eta0, Hi, u0, v0,                 nx, ny,                 dx, dy, dt,                 g, f, r,                 wind_stress=wind,                 boundary_conditions=boundaryConditions,                 use_rk2=True,                 write_netcdf=True)

#Calculate radius from center of bump for plotting
x_center = dx*nx/2.0
y_center = dy*ny/2.0
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))


fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                u0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                v0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] )

#eta1, u1, v1 = sim.download()
    
for i in range(10):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    eta1, hu1, hv1 = sim.download()

    print("i: " + str(i) + " t: " + str(t))

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                u1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                v1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] )

# Close nc-file
sim.cleanUp()

# utility function to print last produced nc-file
nc_files = glob.glob(os.path.join(".", "netcdf_" + datetime.datetime.now().strftime("%Y_%m_%d"), "*"))
if not nc_files:
    raise Exception("No nc-files produced today!")
kp07_nc_files = filter(lambda k: 'KP07' in k, nc_files)
print(kp07_nc_files)
print(max(kp07_nc_files))

nc_files = glob.glob(os.path.join(".", "netcdf_" + datetime.datetime.now().strftime("%Y_%m_%d"), "*"))
if not nc_files:
    raise Exception("No nc-files produced today!")
kp07_nc_files = filter(lambda k: 'KP07' in k, nc_files)
filename =  max(kp07_nc_files)

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()

#Initialize simulator at time time0, using last recorded time step as initial conditions
sim = KP07.KP07.fromfilename(gpu_ctx,                              filename,                              cont_write_netcdf=True)

nx = sim.nx
ny = sim.ny

dx = sim.dx
dy = sim.dy

eta0, hu0, hv0 = sim.download()

#Calculate radius from center of bump for plotting
x_center = dx*nx/2.0
y_center = dy*ny/2.0
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))


fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                u0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                v0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] )


#eta1, u1, v1 = sim.download()
    
for i in range(10):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = sim.t
    eta1, hu1, hv1 = sim.download()

    print("i: " + str(i) + " t: " + str(t))
    
fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                u1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                v1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] )

# Close nc-file
sim.cleanUp()

# DEFINE PARAMETERS

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()
    
#Coriolis well balanced reconstruction scheme
nx = 100
ny = 200

dx = 200.0
dy = 200.0

width = nx * dx
height = ny * dy

dt = 1
g = 9.81
r = 0.0
f = 0.01

wind = WindStress.WindStress()

bcSettings = 1
ghosts = np.array([2,2,2,2]) # north, east, south, west
validDomain = np.array([2,2,2,2])
if (bcSettings == 1):
    boundaryConditions = Common.BoundaryConditions()
    # Wall boundary conditions
elif (bcSettings == 2):
    # periodic boundary conditions
    boundaryConditions = Common.BoundaryConditions(2,2,2,2)
elif bcSettings == 3:
    # periodic north-south
    boundaryConditions = Common.BoundaryConditions(2,1,2,1)
else:
    # periodic east-west
    boundaryConditions = Common.BoundaryConditions(1,2,1,2)
dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * 60;

eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
hu0 = np.zeros(dataShape, dtype=np.float32, order='C');
hv0 = np.zeros(dataShape, dtype=np.float32, order='C');

#Create bump in to lower left of domain for testing
x_center = dx*nx*0.3
y_center = dy*ny*0.2
makeCentralBump(eta0, 0.0, nx, ny, dx, dy, ghosts)
   

#Initialize simulator
sim = CDKLM16.CDKLM16(gpu_ctx, eta0, hu0, hv0, Hi,                       nx, ny, dx, dy, dt, g, f, r,                       boundary_conditions=boundaryConditions,                       write_netcdf=True)

#Calculate radius from center of bump for plotting
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                hu0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                hv0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]])

#eta1, u1, v1 = sim.download()
    
for i in range(10):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    eta1, hu1, hv1 = sim.download()

    print("i: " + str(i) + " t: " + str(t))

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                hu1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                hv1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]])

# Close nc-file
sim.cleanUp()

# utility function to print last produced nc-file
nc_files = glob.glob(os.path.join(".", "netcdf_" + datetime.datetime.now().strftime("%Y_%m_%d"), "*"))
if not nc_files:
    raise Exception("No nc-files produced today!")
cdklm_nc_files = filter(lambda k: 'CDKLM' in k, nc_files)
print(cdklm_nc_files)
print(max(cdklm_nc_files))

nc_files = glob.glob(os.path.join(".", "netcdf_" + datetime.datetime.now().strftime("%Y_%m_%d"), "*"))
if not nc_files:
    raise Exception("No nc-files produced today!")
cdklm_nc_files = filter(lambda k: 'CDKLM' in k, nc_files)
filename =  max(cdklm_nc_files)

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()

#Initialize simulator at time time0, using last recorded time step as initial conditions
sim = CDKLM16.CDKLM16.fromfilename(gpu_ctx,                                    filename,                                    cont_write_netcdf=True)

nx = sim.nx
ny = sim.ny

dx = sim.dx
dy = sim.dy

eta0, hu0, hv0 = sim.download()

#Calculate radius from center of bump for plotting
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_center = dx*nx*0.3
y_center = dy*ny*0.2
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

ghosts = [2,2,2,2]

fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                hu0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                hv0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]])


#eta1, u1, v1 = sim.download()
    
for i in range(10):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = sim.t
    eta1, hu1, hv1 = sim.download()

    print("i: " + str(i) + " t: " + str(t))
    
fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                eta1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                hu1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                hv1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]])

# Close nc-file
sim.cleanUp()


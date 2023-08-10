#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#Import packages we need
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, rc


import os
import pyopencl
import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../')))

# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile

#Finally, import our simulator
from SWESimulators import FBL, CTCS

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

#Finally, import our simulator
from SWESimulators import FBL, CTCS,  KP07, CDKLM16, PlotHelper, Common
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

# Kurganov-Petrova 2007 paper
reload(KP07)
nx = 100
ny = 200

dx = 200.0
dy = 200.0

dt = 0.95
g = 9.81

f = 0.00
r = 0.0
wind = Common.WindStressParams(type=99)


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
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C');

Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')
makeBathymetryCrater(Bi, nx+1, ny+1, dx, dy, ghosts)
#makeBathymetryCrazyness(Bi, nx+1, ny+1, dx, dy, ghosts)
           
#Initialize simulator
reload(KP07)
reload(Common)
sim = KP07.KP07(cl_ctx,                 h0, Bi, u0, v0,                 nx, ny,                 dx, dy, dt,                 g, f, r,                 wind_stress=wind,                 boundary_conditions=boundaryConditions)


#Calculate radius from center of bump for plotting
x_center = dx*nx/2.0
y_center = dy*ny/2.0
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))


fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                h0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] - waterHeight, 
                                u0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                v0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] )

T = 50
def animate(i):
    if (i>0):
        t = sim.step(10.0)
    else:
        t = 0.0
    h1, u1, v1 = sim.download()

    brighten = 1000
    plotter.plot(brighten*(h1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] - waterHeight), 
                 brighten*u1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                 brighten*v1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] )
    fig.suptitle("KP07 Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print "{:03.0f}".format(100.0*i / T) + " % => t=" + str(t)
             
anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
plt.close(anim._fig)
anim

# Printing information on momentum and water height from 
# the above simulatiuon;
h1, u1, v1 = sim.download()
print("At time " + str(sim.t))
print("min-max h1: ", [np.min(h1), np.max(h1)])
print("min-max u1: ", [np.min(u1), np.max(u1)])
print("min-max v1: ", [np.min(v1), np.max(v1)])

# Kurganov-Petrova 2007 paper
reload(KP07)
nx = 7
ny = 7

dx = 200.0
dy = 200.0

dt = 0.95*5
g = 9.81


f = 0.00
r = 0.0

ghosts = np.array([2,2,2,2]) # north, east, south, west
validDomain = np.array([2,2,2,2])
boundaryConditions = Common.BoundaryConditions()
 
dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])
waterHeight = 60
h0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C');

Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')
Bi[6,6] = 30
Bi[6,5] = 30
Bi[5,6] = 30
Bi[5,5] = 30


            
#Initialize simulator
reload(KP07)
reload(PlotHelper)
reload(Common)
sim = KP07.KP07(cl_ctx,                 h0, Bi, u0, v0,                 nx, ny,                 dx, dy, dt,                 g, f, r,                 wind_stress=wind,                 boundary_conditions=boundaryConditions,                 use_rk2=False)


#Calculate radius from center of bump for plotting
x_center = dx*nx/2.0
y_center = dy*ny/2.0
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))


fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                h0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] - waterHeight, 
                                u0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
                                v0[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]],
                               interpolation_type="nearest")


t = sim.step(dt)
    
h1, u1, v1 = sim.download()

brighten = 100000
plotter.plot(brighten*(h1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] - waterHeight), 
             brighten*u1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]], 
             brighten*v1[validDomain[2]:-validDomain[0], validDomain[3]:-validDomain[1]] )
fig.suptitle("KP07 Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

## THIS IS THE DUMMY TEST CELL!
print ("u1:", u1)
print ("v1:", v1)
print(t)
print("min-max h1: ", [np.min(h1), np.max(h1)])
print("min-max u1: ", [np.min(u1), np.max(u1)])
print("min-max v1: ", [np.min(v1), np.max(v1)])
fig2 = plt.figure()
PlotHelper.SinglePlot(fig2, x_coords, y_coords, Bi, interpolation_type="none", title="Bi")
fig3 = plt.figure()
PlotHelper.SinglePlot(fig3, x_coords, y_coords, h1, interpolation_type="none", title="h1")
fig4 = plt.figure()
PlotHelper.SinglePlot(fig4, x_coords, y_coords, np.transpose(u1)-v1, interpolation_type="none", title="np.transpose(u1)-v1")
print("Comparing 2nd and 3rd component: ", np.max(np.fabs(np.transpose(u1)-v1)))
#PlotHelper.SinglePlot(fig4, x_coords, y_coords, u1-v1, interpolation_type="none", title="u1-v1")
#print("Comparing 2nd and 3rd component: ", np.max(np.fabs(u1-v1)))
fig5 = plt.figure()
PlotHelper.SinglePlot(fig5, x_coords, y_coords, u1, interpolation_type="none", title="u1")
fig6 = plt.figure()
PlotHelper.SinglePlot(fig6, x_coords, y_coords, v1, interpolation_type="none", title="v1")

#saveResults(h1, u1, v1, "KP07", "sym_noRestForLake")


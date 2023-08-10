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
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../')))

# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

sys.path.insert(0, '../')

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
    sim.cleanUp()
    
#Coriolis well balanced reconstruction scheme
nx = 30
ny = 60

dx = 200.0
dy = 200.0

dt = 0.95#/5.0
g = 9.81

A = 100.0   # <-- okay with sim.step(24.75*dt)
# A = 1.0     # <-- fails with sim.step(24.75*dt) 

f = 0.00
r = 0.0
wind = Common.WindStressParams(type=99)

# Numerical Sponge
ghosts = [10, 10, 10, 10] # north, east, south, west
boundaryConditions = Common.BoundaryConditions(3,3,3,3, spongeCells=ghosts)
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
                                v0[validDomain[2]+1:validDomain[0], validDomain[3]:validDomain[1]])

T = 120

def animate(i):
    if (i>0):
        #t = sim.step(5.99999*dt)
        #t = sim.step(5*5.0*dt*0.99)
        #n = 25
        #t = sim.step(n*dt - dt*0.1)
        
        #t = sim.step(24.75*dt) # Not ok
        t = sim.step(24.75*dt) # OK when A = 100.
        #t = sim.step(24.975*dt) # OK
        #t = sim.step(24.85*dt) # Barely okay
        
    else:
        t = 0.0
    eta1, u1, v1 = sim.download()   
    brighten = 1 # Increase the values in the animation
    
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

#Clean up old simulator if any:
if 'sim' in globals():
    sim.cleanUp()
    
#CTCS
nx = 200
ny = 200

dx = 200.0
dy = 200.0

g = 9.81

f = 0.02
r = 0.0

# Open boundary 
ghosts = np.array([10,10,10,10]) # north, east, south, west
validDomain = np.array([10,10,10,10])
boundaryConditions = Common.BoundaryConditions(3,3,3,3, spongeCells=ghosts)

#ghosts = np.array([2,2,2,2])
#validDomain = np.array([2,2,2,2])
#boundaryConditions = Common.BoundaryConditions(1,1,1,1)


#Calculate radius from center of bump for plotting
x_center = dx*nx/2.0
y_center = dy*ny/2.0
y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))
min_x = np.min(x_coords[:,0]);
min_y = np.min(y_coords[0,:]);

max_x = np.max(x_coords[0,:]);
max_y = np.max(y_coords[:,0]);



dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

waterHeight = 60


# Staggered u and v
eta0 = np.zeros(dataShape, dtype=np.float32, order='C')
H0 = np.ones(dataShape, dtype=np.float32, order='C') * waterHeight;
u0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32, order='C');
v0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32, order='C');

addCentralBump(eta0, nx, ny, dx, dy, validDomain)
nx = nx
ny = ny

t1 = 250.0
t2 = 850.0
t3 = 2500.0
timeString = " at t = " + str(t1)
mainFig = plt.figure(figsize=(8,7))
figh = plt.subplot(3,1,1)
plt.title("water depth"+timeString)
figu = plt.subplot(3,1,2)
plt.title("water depth at t = " + str(t2))
figShock = plt.subplot(3,1,3)
plt.title("water depth at t = " + str(t3))
plt.tight_layout()

zoomFig = plt.figure(figsize=(8, 6))
plt.title("water depth at t = " + str(t3))



x = np.linspace(0.0, nx*dx, num=nx)
step_size_reductions = [2,4,8]
symbols = [":", "-", "--", "-.", ":", "--", "-." ]
colors = ['k', 'r', 'b', 'g', 'c', 'm']
As = [0, 1.0, 5.0, 10.0, 50.0, 100.0]

h1_2_2 = None
h2_2_2 = None
reload(CDKLM16)
reload(CTCS)

for i in range(6):
    for step_size_reduction in step_size_reductions:

        color = colors[i]
        A = As[i]
        
        dt = 0.95/step_size_reduction
            
        symbol = ''
        if (step_size_reduction == step_size_reductions[1]):
            symbol = ':'
        elif (step_size_reduction == step_size_reductions[2]):
            symbol = '--'
        
        
        print "Starting CTCS with (step_size_reduction, A)",  (step_size_reduction, A)

        sim = CTCS.CTCS(cl_ctx,                    H0, eta0, u0, v0,                    nx, ny,                    dx, dy, dt,                    g, f, r, A,                    boundary_conditions=boundaryConditions)
        label = "A = " + str(A) + ", dt = 0.95/" + str(step_size_reduction)
        
        t = sim.step(t1)
        h1, u1, v1 = sim.download()
        
        plt.figure(mainFig.number)
        
        h = h1[(ny+20)/2, 10:-10]
        u = u1[(ny+20)/2, 10:-11]/(h+waterHeight)
        h = h + waterHeight
        figh = plt.subplot(3,1,1)
        plt.plot(x, h, color+symbol, label=label)
        #figh = plt.subplot(3,1,2)
        #plt.plot(x, u, color+symbol, label=label)
        

        t = sim.step(t2-t1)
        h1, u1, v1 = sim.download()

        h = h1[10:-10, (nx+20)/2]
        h = h + waterHeight
        figh = plt.subplot(3,1,2)
        #if dim_split < 2:
        plt.plot(x, h, color+symbol, label=label)

        t = sim.step(t3-t2)
        h1, u1, v1 = sim.download()

        h = h1[10:-10, (nx+20)/2]
        h = h + waterHeight
        figh = plt.subplot(3,1,3)
        #if dim_split < 2:
        plt.plot(x, h, color+symbol, label=label)

        
        plt.figure(zoomFig.number)

        #if dim_split == 0 and rk_order > 1:
        plt.plot(x[80:-80], h[80:-80], color+symbol, label=label)
        
        #if rk_order == 2 and dim_split == 0:
        #    h1_2_2 = np.copy(h1[10:-10,10:-10])
        #    h2_2_2 = np.copy(h2[10:-10,10:-10])
            
        print "Total amount of execive water: ", np.sum(np.sum(h2-60))
        print "Max water height: ", np.max(h2) + waterHeight
        print " "
            
        sim.cleanUp()
        
plt.figure(mainFig.number)

plt.subplot(3,1,1)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure(zoomFig.number)
plt.ylim(60.01, 60.07)
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)






#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
#%config InlineBackend.figure_format = 'retina'

#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import datetime
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

#Set large figure sizes
#rc('figure', figsize=(16.0, 12.0))
#rc('animation', html='html5')
plt.rcParams["animation.html"] = "jshtml"

#Import our simulator
from SWESimulators import FBL, CTCS, KP07, CDKLM16, PlotHelper, Common, WindStress, IPythonMagic
#Import initial condition and bathymetry generating functions:
from SWESimulators.BathymetryAndICs import *

get_ipython().run_line_magic('setup_logging', '--out compareschemes2d.log')
get_ipython().run_line_magic('cuda_context_handler', 'gpu_ctx')

#Create output directory for images
imgdir='images_' + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
os.makedirs(imgdir)
print("Saving images to " + imgdir)

# Set initial conditions common to all simulators
sim_args = {
"gpu_ctx": gpu_ctx,
"nx": 100, "ny": 200,
"dx": 200.0, "dy": 200.0,
"dt": 1,
"g": 9.81,
"f": 0.0012,
"coriolis_beta": 1.0e-6,
"r": 0.0
}


def sim_animation(simulator, T):
    eta1, u1, v1 = sim.download(interior_domain_only=True)
    
    #Create figure and plot initial conditions
    fig = plt.figure(figsize=(12, 8))
    domain_extent = [0, sim.nx*sim.dx, 0, sim.ny*sim.dy]
    
    ax_eta = plt.subplot(1,3,1)
    sp_eta = ax_eta.imshow(eta1, interpolation="none", origin='bottom', vmin=-1.5, vmax=1.5, extent=domain_extent)
    
    ax_u = plt.subplot(1,3,2)
    sp_u = ax_u.imshow(u1, interpolation="none", origin='bottom', vmin=-1.5, vmax=1.5, extent=domain_extent)
    
    ax_v = plt.subplot(1,3,3)
    sp_v = ax_v.imshow(v1, interpolation="none", origin='bottom', vmin=-1.5, vmax=1.5, extent=domain_extent)
    
    #Helper function which simulates and plots the solution
    def animate(i):
        if (i>0):
            t = sim.step(10.0)
        else:
            t = 0.0
        eta1, u1, v1 = sim.download(interior_domain_only=True)
        
        #Update plots
        fig.sca(ax_eta)
        sp_eta.set_data(eta1)
        
        fig.sca(ax_u)
        sp_u.set_data(u1)
        
        fig.sca(ax_v)
        sp_v.set_data(v1)
        
        fig.suptitle("Time = {:04.0f} s ({:s})".format(t, sim.__class__.__name__), fontsize=18)
        print(".", end='')

    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
    plt.close(fig)
    return anim

ghosts = [0,0,0,0] # north, east, south, west
dataShape = (sim_args["ny"] + ghosts[0]+ghosts[2], 
             sim_args["nx"] + ghosts[1]+ghosts[3])
    
h0 = np.ones(dataShape, dtype=np.float32) * 60
eta0 = np.zeros(dataShape, dtype=np.float32)
u0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32)
v0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32)
    
#Create bump in to lower left of domain for testing
addCentralBump(eta0, sim_args["nx"], sim_args["ny"], sim_args["dx"], sim_args["dy"], ghosts)

#Initialize simulator
fbl_args = {"H": h0, "eta0": eta0, "hu0": u0, "hv0": v0}
sim = FBL.FBL(**fbl_args, **sim_args)

#Run a simulation and plot it
sim_animation(sim, T=50)

ghosts = [1,1,1,1] # north, east, south, west
dataShape = (sim_args["ny"] + ghosts[0]+ghosts[2], 
             sim_args["nx"] + ghosts[1]+ghosts[3])

h0 = np.ones(dataShape, dtype=np.float32) * 60.0;
eta0 = np.zeros(dataShape, dtype=np.float32);
u0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32);
v0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32);       

#Create bump in to lower left of domain for testing
addCentralBump(eta0, sim_args["nx"], sim_args["ny"], sim_args["dx"], sim_args["dy"], ghosts)

#Initialize simulator
ctcs_args = {"H": h0, "eta0": eta0, "hu0": u0, "hv0": v0, "A": 1.0}
sim = CTCS.CTCS(**ctcs_args, **sim_args)

#Run a simulation and plot it
sim_animation(sim, T=50)

ghosts = np.array([2,2,2,2]) # north, east, south, west
dataShape = (sim_args["ny"] + ghosts[0]+ghosts[2], 
             sim_args["nx"] + ghosts[1]+ghosts[3])

H = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32) * 60.0
eta0 = np.zeros(dataShape, dtype=np.float32)
u0 = np.zeros(dataShape, dtype=np.float32)
v0 = np.zeros(dataShape, dtype=np.float32)

#Create bump in to lower left of domain for testing
addCentralBump(eta0, sim_args["nx"], sim_args["ny"], sim_args["dx"], sim_args["dy"], ghosts)

#Initialize simulator
ctcs_args = {"H": H, "eta0": eta0, "hu0": u0, "hv0": v0, "use_rk2": True}
sim = KP07.KP07(**ctcs_args, **sim_args)

#Run a simulation and plot it
sim_animation(sim, T=50)

ghosts = np.array([2,2,2,2]) # north, east, south, west
dataShape = (sim_args["ny"] + ghosts[0]+ghosts[2], 
             sim_args["nx"] + ghosts[1]+ghosts[3])

H = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32) * 60.0
eta0 = np.zeros(dataShape, dtype=np.float32)
u0 = np.zeros(dataShape, dtype=np.float32)
v0 = np.zeros(dataShape, dtype=np.float32)

#Create bump in to lower left of domain for testing
addCentralBump(eta0, sim_args["nx"], sim_args["ny"], sim_args["dx"], sim_args["dy"], ghosts)

#Initialize simulator
ctcs_args = {"H": H, "eta0": eta0, "hu0": u0, "hv0": v0, "rk_order": 2}
sim = CDKLM16.CDKLM16(**ctcs_args, **sim_args)

#Run a simulation and plot it
sim_animation(sim, T=50)

str(cuda.Context.get_cache_config())




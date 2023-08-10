#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt

import os
import pycuda
import pycuda.driver as cuda
import logging
from pycuda.compiler import SourceModule
import datetime
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

#Set large figure sizes
rc('figure', figsize=(8.0, 6.0))
rc('animation', html='html5')

#Import our simulator
from SWESimulators import PlotHelper, Common, WindStress, IPythonMagic
#Import initial condition and bathymetry generating functions:
from SWESimulators.BathymetryAndICs import *

get_ipython().run_line_magic('setup_logging', 'interpolation.log')
get_ipython().run_line_magic('cuda_context_handler', 'gpu_ctx')

# Create data
tex_nx, tex_ny = 3, 2
nx, ny = 50, 50
width, height = 50, 50
dx, dy = np.float32(width/nx), np.float32(height/ny)
sx = np.linspace(1.0, 2.0, tex_nx, dtype=np.float32)
sy = np.linspace(2.0, 3.0, tex_ny, dtype=np.float32)
X = np.outer(sy, sx)
Y = 10 - X*X

plt.figure()
plt.imshow(X, interpolation='none', origin='lower')
plt.colorbar()

plt.figure()
plt.imshow(Y,interpolation='none', origin='lower')
plt.colorbar()

with Common.Timer("Compilation") as t:
    #Compile and get function
    interpolation_module = gpu_ctx.get_kernel("Interpolation.cu")
    
    #Create stream, block, and grid
    stream = cuda.Stream()
    block=(16, 16, 1)
    grid=(int(np.ceil(nx / float(block[0]))), int(np.ceil(ny / float(block[1]))))
    
    
    def setTexture(texref, numpy_array):            
        #Upload data to GPU and bind to texture reference
        texref.set_array(cuda.np_to_array(numpy_array, order="C"))

        # Set texture parameters
        texref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
        texref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
        texref.set_address_mode(1, cuda.address_mode.CLAMP)
        texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing
    
    #Get texture reference from module
    texref_curr = interpolation_module.get_texref("my_texture_current")
    setTexture(texref_curr, X)
    
    texref_next = interpolation_module.get_texref("my_texture_next")
    setTexture(texref_next, Y)
    
    interpolation_kernel = interpolation_module.get_function("interpolationTest")
    interpolation_kernel.prepare("iifffPi", texrefs=[texref_curr, texref_next])
    
    # Allocate output data
    output_gpu = Common.CUDAArray2D(stream, nx, ny, 0, 0, np.zeros((ny, nx), dtype=np.float32))
    
print("Compilation etc took " + str(t.secs))

for t in np.linspace(0.0, 1.0, 4):
    interpolation_kernel.prepared_async_call(grid, block, stream, 
                                             nx, ny,
                                             dx, dy,
                                             t,
                                             output_gpu.data.gpudata, output_gpu.pitch)
    output = output_gpu.download(stream)
    stream.synchronize()
    
    plt.figure()
    plt.imshow(output, interpolation='none', origin='lower')
    plt.colorbar()
    plt.title(str(t))

nx = 100
ny = 200

dx = 200.0
dy = 200.0

dt = 1.0
g = 9.81
f = 0.012
r = 0.0

T = 5

#wind = WindStress.UniformAlongShoreWindStress(tau0=3.0, rho=1025, alpha=1.0/(100*dx)) 
ones = np.ones((4,8), dtype=np.float32)*0.25
t = [2, 3, 4, 10]
X = [ones, ones, -2*ones, -2*ones]
Y = [2*ones, -ones, -ones, 2*ones]
wind = WindStress.WindStress(t, X, Y)

ghosts = [1,1,1,1] # north, east, south, west

dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

h0 = np.ones(dataShape, dtype=np.float32, order='C') * 60;
addTopographyBump(h0, nx, ny, dx, dy, ghosts, 40)

eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
u0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32, order='C');
v0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32, order='C'); 

#Initialize simulator
from SWESimulators.CTCS import CTCS
sim = CTCS(gpu_ctx,               h0, eta0, u0, v0,               nx, ny,               dx, dy, dt,               g, f, r,               wind_stress=wind)
   
for i in range(T):
    t = sim.step(50*dt)
    eta1, u1, v1 = sim.download(interior_domain_only=True)

    plt.figure()
    plt.title("FBL Time = " + "{:04.0f}".format(t) + " s", fontsize=18)
    plt.subplot(1,3,1)
    plt.imshow(eta1)
    plt.subplot(1,3,2)
    plt.imshow(u1)
    plt.subplot(1,3,3)
    plt.imshow(v1)

ghosts = [0,0,0,0] # north, east, south, west

dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

h0 = np.ones(dataShape, dtype=np.float32, order='C') * 60;
addTopographyBump(h0, nx, ny, dx, dy, ghosts, 40)

eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
u0 = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32, order='C');
v0 = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32, order='C'); 

#Initialize simulator
from SWESimulators.FBL import FBL
sim = FBL(gpu_ctx,               h0, eta0, u0, v0,               nx, ny,               dx, dy, dt,               g, f, r,               wind_stress=wind)
   
for i in range(T):
    t = sim.step(50*dt)
    eta1, u1, v1 = sim.download(interior_domain_only=True)

    plt.figure()
    plt.title("FBL Time = " + "{:04.0f}".format(t) + " s", fontsize=18)
    plt.subplot(1,3,1)
    plt.imshow(eta1)
    plt.subplot(1,3,2)
    plt.imshow(u1)
    plt.subplot(1,3,3)
    plt.imshow(v1)

ghosts = [2,2,2,2] # north, east, south, west

dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * 60;
addTopographyBump(Hi, nx, ny, dx, dy, ghosts, 40)

eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C'); 

#Initialize simulator
from SWESimulators.CDKLM16 import CDKLM16
sim = CDKLM16(gpu_ctx,               eta0, u0, v0, Hi,               nx, ny,               dx, dy, dt,               g, f, r,               wind_stress=wind)
   
for i in range(T):
    t = sim.step(50*dt)
    eta1, u1, v1 = sim.download(interior_domain_only=True)

    plt.figure()
    plt.title("FBL Time = " + "{:04.0f}".format(t) + " s", fontsize=18)
    plt.subplot(1,3,1)
    plt.imshow(eta1)
    plt.subplot(1,3,2)
    plt.imshow(u1)
    plt.subplot(1,3,3)
    plt.imshow(v1)

ghosts = [2,2,2,2] # north, east, south, west

dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])

Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * 60;
addTopographyBump(Hi, nx, ny, dx, dy, ghosts, 40)

eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C'); 

#Initialize simulator
from SWESimulators.KP07 import KP07
sim = KP07(gpu_ctx,               eta0, Hi, u0, v0,               nx, ny,               dx, dy, dt,               g, f, r,               wind_stress=wind)
   
for i in range(T):
    t = sim.step(50*dt)
    eta1, u1, v1 = sim.download(interior_domain_only=True)

    plt.figure()
    plt.title("FBL Time = " + "{:04.0f}".format(t) + " s", fontsize=18)
    plt.subplot(1,3,1)
    plt.imshow(eta1)
    plt.subplot(1,3,2)
    plt.imshow(u1)
    plt.subplot(1,3,3)
    plt.imshow(v1)




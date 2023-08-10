#Lets have matplotlib "inline"
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

#Lets have opencl ipython integration enabled
get_ipython().magic('load_ext pyopencl.ipython_ext')

#Import packages we need
import numpy as np
import pyopencl as cl
import os
from matplotlib import animation, rc
from matplotlib import pyplot as plt

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

#Setup easier to use compilation of OpenCL
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_CTX"] = "0"
os.environ["CUDA_CACHE_DISABLE"] = "1"

#Create OpenCL context
cl_ctx = cl.create_some_context()

#Create an OpenCL command queue
cl_queue = cl.CommandQueue(cl_ctx)

get_ipython().run_cell_magic('cl_kernel', '', '__kernel void heat_eq_1D(__global float *u1, __global const float *u0, float kappa, float dt, float dx) {\n    int i = get_global_id(0); //Skip ghost cells\n    int nx = get_global_size(0); //Get total number of cells\n\n    //Internal cells\n    if (i > 0 && i < nx-1) {\n        u1[i] = u0[i] + kappa*dt/(dx*dx) * (u0[i-1] - 2*u0[i] + u0[i+1]);\n    }\n    //Boundary conditions (ghost cells)\n    else { \n        u1[i] = u0[i];\n    }\n}')

"""
Class that holds data for the heat equation in OpenCL
"""
class HeatDataCL:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, u0):
        #Make sure that the data is single precision floating point
        assert(np.issubdtype(u0.dtype, np.float32))
        
        #Find number of cells
        self.nx = len(u0)
        
        mf = cl.mem_flags 
        
        #Upload data to the device
        self.u0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u0)
        
        #Allocate output buffers
        self.u1 = cl.Buffer(cl_ctx, mf.READ_WRITE, u0.nbytes)
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self):
        #Allocate data on the host for result
        u0 = np.empty(self.nx, dtype=np.float32)
        
        #Copy data from device to host
        cl.enqueue_copy(cl_queue, u0, self.u0)
        
        #Return
        return u0;

"""
Computes the heat equation using an explicit finite difference scheme with OpenCL
"""
def opencl_heat_eq(cl_data, kappa, dx, nt):
    #Calculate dt from the CFL condition
    dt = 0.8 * dx*dx / (2.0*kappa)

    #Loop through all the timesteps
    for i in range(nt):
        #Execute program on device
        heat_eq_1D(cl_queue, (cl_data.nx,1), None, cl_data.u1, cl_data.u0, np.float32(kappa), np.float32(dt), np.float32(dx))
        
        #Swap variables
        cl_data.u0, cl_data.u1 = cl_data.u1, cl_data.u0

#Create test input data
u0 = np.random.rand(50).astype(np.float32)
cl_data = HeatDataCL(u0)
kappa = 1.0
dx = 1.0

#Plot initial conditions
plt.figure()
plt.plot(u0, label="u0")

for i in range(1, 5):
    timesteps_per_plot=10
    #Simulate 10 timesteps
    opencl_heat_eq(cl_data, kappa, dx, timesteps_per_plot)

    #Download data
    u1 = cl_data.download()

    #Plot
    plt.plot(u1, label="u"+str(timesteps_per_plot*i))
    
plt.legend()




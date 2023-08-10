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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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

get_ipython().run_cell_magic('cl_kernel', '', '__kernel void heat_eq_2D(__global float *u1, __global const float *u0, float kappa, float dt, float dx, float dy) {\n    //Get total number of cells\n    int nx = get_global_size(0); \n    int ny = get_global_size(1);\n\n    //Get position in grid\n    int i = get_global_id(0); \n    int j = get_global_id(1); \n    int center = j*nx + i;\n\n    //Internal cells\n    if (i > 0 && i < nx-1 && j > 0 && j <ny-1) {\n        //Calculate the four indices of our neighboring cells\n        int north = (j+1)*nx + i;\n        int south = (j-1)*nx + i;\n        int east = j*nx + i+1;\n        int west = j*nx + i-1;\n\n        u1[center] = u0[center]\n            + kappa*dt/(dx*dx) * (u0[west] - 2*u0[center] + u0[east])\n            + kappa*dt/(dy*dy) * (u0[south] - 2*u0[center] + u0[north]);\n    }\n    //Boundary conditions (ghost cells)\n    else { \n        u1[center] = u0[center];\n    }\n}')

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
        assert(not np.isfortran(u0))
        
        #Find number of cells
        self.nx = u0.shape[1]
        self.ny = u0.shape[0]
        
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
        u0 = np.empty((self.ny, self.nx), dtype=np.float32)
        
        #Copy data from device to host
        cl.enqueue_copy(cl_queue, u0, self.u0) 
        
        #Return
        return u0;

"""
Computes the heat equation using an explicit finite difference scheme with OpenCL
"""
def opencl_heat_eq(cl_data, kappa, dx, dy, nt):
    #Calculate dt from the CFL condition
    dt = 0.4 * min(dx*dx / (2.0*kappa), dy*dy / (2.0*kappa))

    #Loop through all the timesteps
    for i in range(0, nt):
        #Execute program on device
        heat_eq_2D(cl_queue, (cl_data.nx, cl_data.ny), None,                    cl_data.u1, cl_data.u0,                    np.float32(kappa), np.float32(dt), np.float32(dx), np.float32(dy))
        
        #Swap variables
        cl_data.u0, cl_data.u1 = cl_data.u1, cl_data.u0

#Create test input data
u0 = np.random.rand(25, 50).astype(np.float32)
cl_data = HeatDataCL(u0)
kappa = 1.0
dx = 1.0
dy = 2.0

#Plot initial conditions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("Heat equation 2D", fontsize=18)

max_x = cl_data.nx*dx
max_y = cl_data.ny*dy

y, x = np.mgrid[0:max_y:dy, 0:max_x:dx]
surf_args=dict(cmap=cm.coolwarm, shade=True, vmin=0.0, vmax=1.0, cstride=1, rstride=1)
ax.plot_surface(x, y, u0, **surf_args)
ax.set_zlim(0.0, 1.0)

def animate(i):
    timesteps_per_plot=1
    
    if (i>0):
        opencl_heat_eq(cl_data, kappa, dx, dy, timesteps_per_plot)
        
    u1 = cl_data.download()
    ax.clear()
    ax.plot_surface(x, y, u1, **surf_args)
            
anim = animation.FuncAnimation(fig, animate, range(50), interval=100)
plt.close(anim._fig)
anim




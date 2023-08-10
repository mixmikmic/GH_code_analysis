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

get_ipython().run_cell_magic('cl_kernel', '', '__kernel void linear_wave_2D(__global float* u2, global const float* u1, __global const float* u0, float kappa, float dt, float dx, float dy) {\n    //Get total number of cells\n    int nx = get_global_size(0); \n    int ny = get_global_size(1);\n\n    //Get position in grid\n    int i = get_global_id(0); \n    int j = get_global_id(1); \n\n    //Calculate the four indices of our neighboring cells\n    int center = j*nx + i;\n    int north = (j+1)*nx + i;\n    int south = (j-1)*nx + i;\n    int east = j*nx + i+1;\n    int west = j*nx + i-1;\n\n    //Internal cells\n    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {\n        u2[center] = 2.0f*u1[center] - u0[center]\n            + kappa*dt/(dx*dx) * (u1[west] - 2*u1[center] + u1[east])\n            + kappa*dt/(dy*dy) * (u1[south] - 2*u1[center] + u1[north]);\n    }\n}\n\n__kernel void linear_wave_2D_bc(__global float* u) {\n    //Get total number of cells\n    int nx = get_global_size(0); \n    int ny = get_global_size(1);\n\n    //Get position in grid\n    int i = get_global_id(0); \n    int j = get_global_id(1); \n\n    //Calculate the four indices of our neighboring cells\n    int center = j*nx + i;\n    int north = (j+1)*nx + i;\n    int south = (j-1)*nx + i;\n    int east = j*nx + i+1;\n    int west = j*nx + i-1;\n\n    if (i == 0) {\n        u[center] = u[east];\n    }\n    else if (i == nx-1) {\n        u[center] = u[west];\n    }\n    else if (j == 0) {\n        u[center] = u[north];\n    }\n    else if (j == ny-1) {\n        u[center] = u[south];\n    }\n}')

"""
Class that holds data for the heat equation in OpenCL
"""
class LinearWaveDataCL:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, u0, u1):
        #Make sure that the data is single precision floating point
        assert(np.issubdtype(u1.dtype, np.float32))
        assert(np.issubdtype(u0.dtype, np.float32))
        assert(not np.isfortran(u0))
        assert(not np.isfortran(u1))
        assert(u0.shape == u1.shape)

        #Find number of cells
        self.nx = u0.shape[1]
        self.ny = u0.shape[0]
        
        mf = cl.mem_flags 
        
        #Upload data to the device
        self.u0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u0)
        self.u1 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u1)
        
        #Allocate output buffers
        self.u2 = cl.Buffer(cl_ctx, mf.READ_WRITE, u0.nbytes)
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self):
        #Allocate data on the host for result
        u1 = np.empty((self.ny, self.nx), dtype=np.float32)
        
        #Copy data from device to host
        cl.enqueue_copy(cl_queue, u1, self.u1)
        
        #Return
        return u1;

"""
Computes a solution to the linear wave equation equation using an explicit finite difference scheme with OpenCL
"""
def opencl_linear_wave(cl_data, c, dx, dy, dt, nt):
    #Loop through all the timesteps
    for i in range(0, nt):
        #Execute program on device
        linear_wave_2D(cl_queue, (nx, ny), None,                        cl_data.u2, cl_data.u1, cl_data.u0,                        np.float32(c), np.float32(dt), np.float32(dx), np.float32(dy))
        linear_wave_2D_bc(cl_queue, (nx, ny), None, cl_data.u2)
        
        #Swap variables
        cl_data.u0, cl_data.u1, cl_data.u2 = cl_data.u1, cl_data.u2, cl_data.u0

#Create test input data
c = 1.0
nx, ny = 50, 25
dx = 1.0
dy = 2.0
dt = 0.1 * min(dx / (2.0*c), dy / (2.0*c))

u0 = np.zeros((ny, nx)).astype(np.float32)
for j in range(ny):
    for i in range(nx):
        x = (i - nx/2.0) * dx
        y = (j - ny/2.0) * dy
        if (np.sqrt(x**2 + y**2) < 10*min(dx, dy)):
            u0[j, i] = 1.0
u1 = u0
cl_data = LinearWaveDataCL(u0, u1)


#Plot initial conditions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("Wave equation 2D", fontsize=18)

max_x = cl_data.nx*dx
max_y = cl_data.ny*dy

y, x = np.mgrid[0:max_y:dy, 0:max_x:dx]
surf_args=dict(cmap=cm.coolwarm, shade=True, vmin=0.0, vmax=1.0, cstride=1, rstride=1)
ax.plot_surface(x, y, u0, **surf_args)
ax.set_zlim(-0.5, 5.0)

    
def animate(i):
    timesteps_per_plot=5
    
    #Simulate
    if (i>0):
        opencl_linear_wave(cl_data, c, dx, dy, dt, timesteps_per_plot)

    #Download data
    u1 = cl_data.download()

    #Plot
    ax.clear()
    ax.plot_surface(x, y, u1, **surf_args)
    ax.set_zlim(-5.0, 5.0)
        

anim = animation.FuncAnimation(fig, animate, range(50), interval=100)
plt.close(anim._fig)
anim




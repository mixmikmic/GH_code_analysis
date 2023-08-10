#Lets have matplotlib "inline"
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

#Lets have opencl ipython integration enabled
get_ipython().magic('load_ext pyopencl.ipython_ext')

#Import packages we need
import numpy as np
import pyopencl as cl
import os
import time
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

print("Using ", cl_ctx.devices[0].name)

#Create an OpenCL command queue
cl_queue = cl.CommandQueue(cl_ctx)

get_ipython().run_cell_magic('cl_kernel', '', '\nfloat3 F(const float3 Q, const float g) {\n    float3 F;\n\n    F.x = Q.y;                              //hu\n    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;\n    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;\n\n    return F;\n}\n\nfloat3 G(const float3 Q, const float g) {\n    float3 G;\n\n    G.x = Q.z;                              //hv\n    G.y = Q.y*Q.z / Q.x;                    //hu*hv/h;\n    G.z = Q.z*Q.z / Q.x + 0.5f*g*Q.x*Q.x;   //hv*hv/h + 0.5f*g*h*h;\n\n    return G;\n}\n\n__kernel void swe_2D(\n        __global float* h1, __global float* hu1, __global float* hv1,\n        __global const float *h0, __global const float *hu0, __global const float *hv0,\n        float g,\n        float dt, float dx, float dy) {\n\n    //Get total number of cells\n    int nx = get_global_size(0); \n    int ny = get_global_size(1);\n\n    //Get position in grid\n    int i = get_global_id(0); \n    int j = get_global_id(1);\n\n    //Internal cells\n    if (i > 0 && i < nx-1 && j > 0 && j <ny-1) {\n        //Calculate the four indices of our neighboring cells\n        int i = get_global_id(0); \n        int j = get_global_id(1);\n\n        int center = j*nx + i;\n        int north = (j+1)*nx + i;\n        int south = (j-1)*nx + i;\n        int east = j*nx + i+1;\n        int west = j*nx + i-1;\n\n        const float3 Q_east = (float3)(h0[east], hu0[east], hv0[east]);\n        const float3 Q_west = (float3)(h0[west], hu0[west], hv0[west]);\n        const float3 Q_north = (float3)(h0[north], hu0[north], hv0[north]);\n        const float3 Q_south = (float3)(h0[south], hu0[south], hv0[south]);\n\n        const float3 F_east = F(Q_east, g);\n        const float3 F_west = F(Q_west, g);\n        const float3 G_north = G(Q_north, g);\n        const float3 G_south = G(Q_south, g);\n\n        float3 Q1 = 0.25f*(Q_east + Q_west + Q_north + Q_south)\n            - dt/(2.0f*dx)*(F_east - F_west)\n            - dt/(2.0f*dy)*(G_north - G_south);\n\n        h1[center] = Q1.x;\n        hu1[center] = Q1.y;\n        hv1[center] = Q1.z;\n    }\n}')

get_ipython().run_cell_magic('cl_kernel', '', '__kernel void swe_2D_bc(__global float* h, __global float* hu, __global float* hv) {\n    //Get total number of cells\n    int nx = get_global_size(0); \n    int ny = get_global_size(1);\n\n    //Get position in grid\n    int i = get_global_id(0); \n    int j = get_global_id(1); \n\n    //Calculate the four indices of our neighboring cells\n    int center = j*nx + i;\n    int north = (j+1)*nx + i;\n    int south = (j-1)*nx + i;\n    int east = j*nx + i+1;\n    int west = j*nx + i-1;\n\n    if (i == 0) {\n        h[center] = h[east];\n        hu[center] = -hu[east];\n        hv[center] = hv[east];\n    }\n    else if (i == nx-1) {\n        h[center] = h[west];\n        hu[center] = -hu[west];\n        hv[center] = hv[west];\n    }\n    else if (j == 0) {\n        h[center] = h[north];\n        hu[center] = hu[north];\n        hv[center] = -hv[north];\n    }\n    else if (j == ny-1) {\n        h[center] = h[south];\n        hu[center] = hu[south];\n        hv[center] = -hv[south];\n    }\n}')

"""
Class that holds data for the heat equation in OpenCL
"""
class SWEDataCL:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, h0, hu0, hv0):
        #Make sure that the data is single precision floating point
        assert(np.issubdtype(h0.dtype, np.float32))
        assert(np.issubdtype(hu0.dtype, np.float32))
        assert(np.issubdtype(hv0.dtype, np.float32))

        assert(not np.isfortran(h0))
        assert(not np.isfortran(hu0))
        assert(not np.isfortran(hv0))

        assert(h0.shape == hu0.shape)
        assert(h0.shape == hv0.shape)

        self.nx = h0.shape[1]
        self.ny = h0.shape[0]

        #Upload data to the device
        mf = cl.mem_flags
        self.h0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h0)
        self.hu0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hu0)
        self.hv0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hv0)

        self.h1 = cl.Buffer(cl_ctx, mf.READ_WRITE, h0.nbytes)
        self.hu1 = cl.Buffer(cl_ctx, mf.READ_WRITE, h0.nbytes)
        self.hv1 = cl.Buffer(cl_ctx, mf.READ_WRITE, h0.nbytes)
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self):
        #Allocate data on the host for result
        h1 = np.empty((self.ny, self.nx), dtype=np.float32)
        hu1 = np.empty((self.ny, self.nx), dtype=np.float32)
        hv1 = np.empty((self.ny, self.nx), dtype=np.float32)
        
        #Copy data from device to host
        cl.enqueue_copy(cl_queue, h1, self.h0)
        cl.enqueue_copy(cl_queue, hu1, self.hu0)
        cl.enqueue_copy(cl_queue, hv1, self.hv0)
        
        #Return
        return h1, hu1, hv1;

"""
Computes a solution to the shallow water equations using an explicit finite difference scheme with OpenCL
"""
def opencl_swe(cl_data, g, dx, dy, dt, nt):

    #Loop through all the timesteps
    for i in range(0, nt):
        #Execute program on device
        swe_2D(cl_queue, (cl_data.nx, cl_data.ny), None,                cl_data.h1, cl_data.hu1, cl_data.hv1,                cl_data.h0, cl_data.hu0, cl_data.hv0,
               np.float32(g), np.float32(dt), np.float32(dx), np.float32(dy))
        swe_2D_bc(cl_queue, (cl_data.nx, cl_data.ny), None, cl_data.h1, cl_data.hu1, cl_data.hv1)
        
        #Swap variables
        cl_data.h0, cl_data.h1 = cl_data.h1, cl_data.h0
        cl_data.hu0, cl_data.hu1 = cl_data.hu1, cl_data.hu0
        cl_data.hv0, cl_data.hv1 = cl_data.hv1, cl_data.hv0

def circular_dambreak_initial_conditions(nx, ny):
    dx = 100.0 / float(nx)
    dy = 100.0 / float(ny)
    dt = 0.05*min(dx, dy) #Estimate of dt that will not violate the CFL condition

    h0 = np.ones((ny, nx), dtype=np.float32);
    hu0 = np.zeros((ny, nx), dtype=np.float32);
    hv0 = np.zeros((ny, nx), dtype=np.float32);

    for j in range(ny):
        for i in range(nx):
            x = (i - nx/2.0) * dx
            y = (j - ny/2.0) * dy
            if (np.sqrt(x**2 + y**2) < 10*min(dx, dy)):
                h0[j, i] = 10.0

    cl_data = SWEDataCL(h0, hu0, hv0)
    return cl_data, dx, dy, dt

#Create test input data
cl_data, dx, dy, dt = circular_dambreak_initial_conditions(64, 64)
g = 9.80665

#Plot initial conditions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("Wave equation 2D", fontsize=18)

max_x = cl_data.nx*dx
max_y = cl_data.ny*dy

y, x = np.mgrid[0:max_y:dy, 0:max_x:dx]
surf_args=dict(cmap=cm.coolwarm, shade=True, vmin=0.0, vmax=1.0, cstride=1, rstride=1)


def animate(i):
    timesteps_per_plot=10

    #Simulate
    if (i>0):
        opencl_swe(cl_data, g, dx, dy, dt, timesteps_per_plot)

    #Download data
    h1, hu1, hv1 = cl_data.download()

    #Plot
    ax.clear()
    ax.plot_surface(x, y, h1, **surf_args)
    ax.set_zlim(-5.0, 5.0)

    
anim = animation.FuncAnimation(fig, animate, range(50), interval=100)
plt.close(anim._fig)
anim




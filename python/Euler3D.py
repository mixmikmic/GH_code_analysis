#Force correct backend of mayavi
import os
os.environ["ETS_TOOLKIT"] = "qt4"

#Lets have matplotlib "inline"
get_ipython().magic('pylab inline')

#Lets have opencl ipython integration enabled
get_ipython().magic('load_ext pyopencl.ipython_ext')

#Lets have large and high-res figures
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

#Import packages we need
import time
import numpy as np
import pyopencl as cl #OpenCL in Python
from mayavi import mlab #mayavi for visualization
import IPython.display as IPdisplay # For movie creation
import tempfile #To get a temporary filename
import matplotlib.animation as manimation

#Try to make mayavi use offscreen rendering
mlab.options.offscreen = True

#Make sure we get compiler output from OpenCL
import os
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

#Set which CL device to use
os.environ["PYOPENCL_CTX"] = "1"

#Create OpenCL context
cl_ctx = cl.create_some_context()

print "Using ", cl_ctx.devices[0].name

#Create an OpenCL command queue
cl_queue = cl.CommandQueue(cl_ctx)

get_ipython().run_cell_magic('cl_kernel', '', '\n/**\n  * Mote that we have to use float8 instead of float5 in OpenCL\n  */\n\n\n\nfloat pressure(float8 q, float gamma) {\n    float rho = q.s0;\n    float u = q.s1;\n    float v = q.s2;\n    float w = q.s3;\n    float E =  q.s4;\n\n    return (gamma-1.0f)*(E-0.5f*(u*u + v*v + w*w)/rho);\n}\n\n\nfloat8 F(float8 q, float gamma) {\n    float rho   = q.s0;\n    float rho_u = q.s1;\n    float rho_v = q.s2;\n    float rho_w = q.s3;\n    float E     =  q.s4;\n\n    float u = rho_u/rho;\n    float P = pressure(q, gamma);\n\n    float8 F;\n\n    F.s0 = rho_u;\n    F.s1 = rho_u*u + P;\n    F.s2 = rho_v*u;\n    F.s3 = rho_w*u;\n    F.s4 = u*(E+P);\n\n    return F;\n}\n\n\n\nfloat8 G(float8 q, float gamma) {\n    float rho   = q.s0;\n    float rho_u = q.s1;\n    float rho_v = q.s2;\n    float rho_w = q.s3;\n    float E     =  q.s4;\n\n    float v = rho_v/rho;\n    float P = pressure(q, gamma);\n\n    float8 G;\n\n    G.s0 = rho_v;\n    G.s1 = rho_u*v;\n    G.s2 = rho_v*v + P;\n    G.s3 = rho_w*v;\n    G.s4 = v*(E+P);\n\n    return G;\n}\n\n\n\nfloat8 H(float8 q, float gamma) {\n    float rho   = q.s0;\n    float rho_u = q.s1;\n    float rho_v = q.s2;\n    float rho_w = q.s3;\n    float E     =  q.s4;\n\n    float w = rho_w/rho;\n    float P = pressure(q, gamma);\n\n    float8 H;\n\n    H.s0 = rho_w;\n    H.s1 = rho_u*w;\n    H.s2 = rho_v*w;\n    H.s3 = rho_w*w + P;\n    H.s4 = w*(E+P);\n\n    return H;\n}\n\n\n\n__kernel void euler_3D(\n        __global float* rho1, __global float* rho_u1, __global float* rho_v1,\n        __global float* rho_w1, __global float* E1,\n        __global float* rho0, __global float* rho_u0, __global float* rho_v0,\n        __global float* rho_w0, __global float* E0,\n        float dt, float gamma,\n        float dx, float dy, float dz) {\n\n\n    //Get total number of cells\n    int nx = get_global_size(0); \n    int ny = get_global_size(1); \n    int nz = get_global_size(2);\n\n    //Get position in grid\n    int i = get_global_id(0); \n    int j = get_global_id(1);\n    int k = get_global_id(2);\n\n    //Internal cells\n    if (       (i > 0 && i < nx-1) \n            && (j > 0 && j < ny-1)\n            && (k > 0 && k < nz-1)) {\n        //Calculate the indices of our neighboring cells\n        int i = get_global_id(0); \n        int j = get_global_id(1); \n        int k = get_global_id(2);\n\n        int center =    k*(nx*ny) + j*nx + i;\n        int x_pos =     k*(nx*ny) +     j*nx + i+1;\n        int x_neg =     k*(nx*ny) +     j*nx + i-1;\n        int y_pos =     k*(nx*ny) + (j+1)*nx + i;\n        int y_neg =     k*(nx*ny) + (j-1)*nx + i;\n        int z_pos = (k+1)*(nx*ny) +     j*nx + i; \n        int z_neg = (k-1)*(nx*ny) +     j*nx + i; \n\n        const float8 Q_x_pos = (float8)(rho0[x_pos], rho_u0[x_pos], rho_v0[x_pos], rho_w0[x_pos], E0[x_pos], 0, 0, 0);\n        const float8 Q_y_pos = (float8)(rho0[y_pos], rho_u0[y_pos], rho_v0[y_pos], rho_w0[y_pos], E0[y_pos], 0, 0, 0);\n        const float8 Q_z_pos = (float8)(rho0[z_pos], rho_u0[z_pos], rho_v0[z_pos], rho_w0[z_pos], E0[z_pos], 0, 0, 0);\n\n        const float8 Q_x_neg = (float8)(rho0[x_neg], rho_u0[x_neg], rho_v0[x_neg], rho_w0[x_neg], E0[x_neg], 0, 0, 0);\n        const float8 Q_y_neg = (float8)(rho0[y_neg], rho_u0[y_neg], rho_v0[y_neg], rho_w0[y_neg], E0[y_neg], 0, 0, 0);\n        const float8 Q_z_neg = (float8)(rho0[z_neg], rho_u0[z_neg], rho_v0[z_neg], rho_w0[z_neg], E0[z_neg], 0, 0, 0);\n\n        //Calculate fluxes\n        const float8 F_x_pos = F(Q_x_pos, gamma);\n        const float8 G_y_pos = G(Q_y_pos, gamma);\n        const float8 H_z_pos = H(Q_z_pos, gamma);\n\n        const float8 F_x_neg = F(Q_x_neg, gamma);\n        const float8 G_y_neg = G(Q_y_neg, gamma);\n        const float8 H_z_neg = H(Q_z_neg, gamma);\n\n        float8 Q1 = 0.1666666666f*(Q_x_pos + Q_x_neg + Q_y_pos + Q_y_neg + Q_z_pos + Q_z_neg)\n            - dt/(2.0f*dx)*(F_x_pos - F_x_neg)\n            - dt/(2.0f*dy)*(G_y_pos - G_y_neg)\n            - dt/(2.0f*dz)*(H_z_pos - H_z_neg);\n\n        rho1[center] = Q1.s0;\n        rho_u1[center] = Q1.s1;\n        rho_v1[center] = Q1.s2;\n        rho_w1[center] = Q1.s3;\n        E1[center] = Q1.s4;\n    }\n}')

get_ipython().run_cell_magic('cl_kernel', '', '__kernel void euler_3D_bc(__global float* rho, __global float* rho_u, __global float* rho_v, __global float* rho_w, __global float* E) {\n    //Get total number of cells\n    int nx = get_global_size(0); \n    int ny = get_global_size(1); \n    int nz = get_global_size(2);\n\n    //Get position in grid\n    int i = get_global_id(0); \n    int j = get_global_id(1); \n    int k = get_global_id(2); \n\n    //Calculate the indices of our neighboring cells\n    int center =    k*(nx*ny) + j*nx + i;\n    int x_pos =     k*(nx*ny) +     j*nx + i+1;\n    int x_neg =     k*(nx*ny) +     j*nx + i-1;\n    int y_pos =     k*(nx*ny) + (j+1)*nx + i;\n    int y_neg =     k*(nx*ny) + (j-1)*nx + i;\n    int z_pos = (k+1)*(nx*ny) +     j*nx + i; \n    int z_neg = (k-1)*(nx*ny) +     j*nx + i; \n\n    if (i == 0) {\n        rho[center] = rho[x_pos];\n        rho_u[center] = -rho_u[x_pos];\n        rho_v[center] = rho_v[x_pos];\n        rho_w[center] = rho_w[x_pos];\n        E[center] = E[x_pos];\n    }\n    else if (i == nx-1) {\n        rho[center] = rho[x_neg];\n        rho_u[center] = -rho_u[x_neg];\n        rho_v[center] = rho_v[x_neg];\n        rho_w[center] = rho_w[x_neg];\n        E[center] = E[x_neg];\n    }\n\n    if (j == 0) {\n        rho[center] = rho[y_pos];\n        rho_u[center] = rho_u[y_pos];\n        rho_v[center] = -rho_v[y_pos];\n        rho_w[center] = rho_w[y_pos];\n        E[center] = E[y_pos];\n    }\n    else if (j == ny-1) {\n        rho[center] = rho[y_neg];\n        rho_u[center] = rho_u[y_neg];\n        rho_v[center] = -rho_v[y_neg];\n        rho_w[center] = rho_w[y_neg];\n        E[center] = E[y_neg];\n    }\n\n    if (k == 0) {\n        rho[center] = rho[z_pos];\n        rho_u[center] = rho_u[z_pos];\n        rho_v[center] = rho_v[z_pos];\n        rho_w[center] = -rho_w[z_pos];\n        E[center] = E[z_pos];\n    }\n    else if (k == nz-1) {\n        rho[center] = rho[z_neg];\n        rho_u[center] = rho_u[z_neg];\n        rho_v[center] = rho_v[z_neg];\n        rho_w[center] = -rho_w[z_neg];\n        E[center] = E[z_neg];\n    }\n}')

"""
Class that holds data for the heat equation in OpenCL
"""
class EulerDataCL:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, rho0, rho_u0, rho_v0, rho_w0, E0):
        #Make sure that the data is single precision floating point
        assert(np.issubdtype(rho0.dtype, np.float32))
        assert(np.issubdtype(rho_u0.dtype, np.float32))
        assert(np.issubdtype(rho_v0.dtype, np.float32))
        assert(np.issubdtype(rho_w0.dtype, np.float32))
        assert(np.issubdtype(E0.dtype, np.float32))

        assert(np.isfortran(rho0))
        assert(np.isfortran(rho_u0))
        assert(np.isfortran(rho_v0))
        assert(np.isfortran(rho_w0))
        assert(np.isfortran(E0))

        assert(rho0.shape == rho_u0.shape)
        assert(rho0.shape == rho_v0.shape)
        assert(rho0.shape == rho_w0.shape)
        assert(rho0.shape == E0.shape)

        #Note that we skip ghost cells
        self.nx = rho0.shape[0]
        self.ny = rho0.shape[1]
        self.nz = rho0.shape[2]

        #Upload data to the device
        mf = cl.mem_flags
        self.rho0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho0)
        self.rho_u0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho_u0)
        self.rho_v0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho_v0)
        self.rho_w0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho_w0)
        self.E0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=E0)

        #Allocate output data
        self.rho1 = cl.Buffer(cl_ctx, mf.READ_WRITE, rho0.nbytes)
        self.rho_u1 = cl.Buffer(cl_ctx, mf.READ_WRITE, rho0.nbytes)
        self.rho_v1 = cl.Buffer(cl_ctx, mf.READ_WRITE, rho0.nbytes)
        self.rho_w1 = cl.Buffer(cl_ctx, mf.READ_WRITE, rho0.nbytes)
        self.E1 = cl.Buffer(cl_ctx, mf.READ_WRITE, rho0.nbytes)
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self):
        #Allocate data on the host for result
        rho1 = np.empty((self.nx, self.ny, self.nz), dtype=np.float32, order='F')
        rho_u1 = np.empty((self.nx, self.ny, self.nz), dtype=np.float32, order='F')
        rho_v1 = np.empty((self.nx, self.ny, self.nz), dtype=np.float32, order='F')
        rho_w1 = np.empty((self.nx, self.ny, self.nz), dtype=np.float32, order='F')
        E1 = np.empty((self.nx, self.ny, self.nz), dtype=np.float32, order='F')
        
        #Copy data from device to host
        cl.enqueue_copy(cl_queue, rho1, self.rho0)
        cl.enqueue_copy(cl_queue, rho_u1, self.rho_u0)
        cl.enqueue_copy(cl_queue, rho_v1, self.rho_v0)
        cl.enqueue_copy(cl_queue, rho_w1, self.rho_w0)
        cl.enqueue_copy(cl_queue, E1, self.E0)
        
        #Return
        return rho1, rho_u1, rho_v1, rho_w1, E1;

"""
Computes a solution to the shallow water equations using an explicit finite difference scheme with OpenCL
"""
def opencl_euler(cl_data, dx, dy, dz, dt, gamma, nt):

    #Loop through all the timesteps
    for i in range(0, nt):
        #Execute program on device
        euler_3D(cl_queue, (cl_data.nx, cl_data.ny, cl_data.nz), None,                cl_data.rho1, cl_data.rho_u1, cl_data.rho_v1, cl_data.rho_w1, cl_data.E1,                cl_data.rho0, cl_data.rho_u0, cl_data.rho_v0, cl_data.rho_w0, cl_data.E0,                np.float32(dt), np.float32(gamma),                np.float32(dx), np.float32(dy), np.float32(dz))
        euler_3D_bc(cl_queue, (cl_data.nx, cl_data.ny, cl_data.nz), None,                     cl_data.rho1, cl_data.rho_u1, cl_data.rho_v1, cl_data.rho_w1, cl_data.E1)
        
        #Swap variables
        cl_data.rho0, cl_data.rho1 = cl_data.rho1, cl_data.rho0
        cl_data.rho_u0, cl_data.rho_u1 = cl_data.rho_u1, cl_data.rho_u0
        cl_data.rho_v0, cl_data.rho_v1 = cl_data.rho_v1, cl_data.rho_v0
        cl_data.rho_w0, cl_data.rho_w1 = cl_data.rho_w1, cl_data.rho_w0
        cl_data.E0, cl_data.E1 = cl_data.E1, cl_data.E0

#Plotting helper
def easy_surf(u, dx, dy, z_max):
    nx = u.shape[0]
    ny = u.shape[1]
    
    max_x = nx*dx
    max_y = ny*dy
    
    x, y = numpy.mgrid[0:max_x:dx, 0:max_y:dy]
    
    mlab.surf(x, y, u, vmin=0, vmax=z_max)
    mlab.axes(extent=[0, max_x, 0, max_y, 0, z_max])
    mlab.view(azimuth=45, elevation=60, distance=3*max(max_x, max_y), focalpoint=(max_x/2.0, max_y/2.0, z_max/2.0))
    scr = mlab.screenshot();
    mlab.close()
    return scr;

#plotting helper
def easy_vol(q, dx, dy, dz, v_max):
    nx = q.shape[0]
    ny = q.shape[1]
    nz = q.shape[2]
    
    max_x = nx*dx
    max_y = ny*dy
    max_z = nz*dz
    
    x, y, z = numpy.mgrid[0:max_x:dx, 0:max_y:dy, 0:max_z:dz]
    
    #Schlieren type visualization
    grad = np.gradient(q)
    color = abs(grad[0]) + abs(grad[1]) + abs(grad[2]) / v_max
    color = np.clip(color, 0.0, 1.0)
        
    voxel_grid = mlab.pipeline.scalar_field(x, y, z, color);
    mlab.pipeline.volume(voxel_grid, vmin=0, vmax=1)
    mlab.axes()
    #mlab.outline()
    mlab.axes(extent=[0, max_x, 0, max_y, 0, max_z])
    mlab.view(azimuth=45, elevation=60, distance=3*max(max_x, max_y), focalpoint=(max_x/2.0, max_y/2.0, max_z/2.0))
    scr = mlab.screenshot();
    mlab.close()
    return scr;

def circular_dambreak_initial_conditions(nx, ny, nz, dx, dy, dz, gamma):
    rho0 = np.ones((nx, ny, nz), dtype=np.float32, order='F');
    rho_u0 = np.zeros((nx, ny, nz), dtype=np.float32, order='F');
    rho_v0 = np.zeros((nx, ny, nz), dtype=np.float32, order='F');
    rho_w0 = np.zeros((nx, ny, nz), dtype=np.float32, order='F');
    E0 = np.zeros((nx, ny, nz), dtype=np.float32, order='F');

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = (i - nx/2.0) * dx
                y = (j - ny/2.0) * dy
                z = (k - nz/2.0) * dz
                if (sqrt(x**2 + z**2) < 10*min([dx, dy, dz])):
                    rho = 5
                    u = 0
                    v = 0
                    w = 0
                    P = 5
                    
                    rho0[i, j, k] = rho
                    E0[i, j, k] = 0.5*rho*(u*u+v*v+w*w)+P/(gamma-1.0)
                else:
                    rho = 1
                    u = 0
                    v = 0
                    w = 0
                    P = 1
                    
                    rho0[i, j, k] = rho
                    E0[i, j, k] = 0.5*rho*(u*u+v*v+w*w)+P/(gamma-1.0)

    cl_data = EulerDataCL(rho0, rho_u0, rho_v0, rho_w0, E0)
    return cl_data

def embed_video(fname, mimetype):
    from IPython.display import HTML
    video_encoded = open(fname, "rb").read().encode("base64")
    video_tag = '<video controls alt="test" src="data:video/{0};base64,{1}">'.format(mimetype, video_encoded)
    return HTML(data=video_tag)

#Setup movie stuff
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Euler Equations in 3D', artist='Matplotlib',
        comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata, codec='libvpx', bitrate=1024*16)

#Create test input data
nx = 64
ny = 64
nz = 64
gamma = 1.4
dx = 100.0 / float(nx)
dy = 100.0 / float(ny)
dz = 100.0 / float(ny)
dt = 0.25*min(dx, dy, dz) #Estimate of dt that will not violate the CFL condition

cl_data = circular_dambreak_initial_conditions(nx, ny, nz, dx, dy, dz, gamma)

fig = figure()

temp_filename = next(tempfile._get_candidate_names()) + ".webm"
with writer.saving(fig, temp_filename, 100):
    #Show initial conditions
    rho1, rho_u1, rho_v1, rho_w1, E1 = cl_data.download()
    im = imshow(easy_vol(rho1[1:-1,1:-1,1:-1], dx, dy, dz, 5.0))
    axis('off')
    writer.grab_frame()

    #Simulate and animate
    max_iter = 70
    start_time = time.time()
    for i in range(0, max_iter):
        timesteps_per_plot=5
        
        #Simulate
        opencl_euler(cl_data, dx, dy, dz, dt, gamma, timesteps_per_plot)

        #Download data
        rho1, rho_u1, rho_v1, rho_w1, E1 = cl_data.download()
        
        #Plot
        im.set_data(easy_vol(rho1[1:-1,1:-1,1:-1], dx, dy, dz, 0.4))
        axis('off')
        writer.grab_frame()
        
        print str(100*i/max_iter) + '% in ' + str(time.time() - start_time) + ' seconds'
        
fig.clear()
embed_video(temp_filename, 'webm')






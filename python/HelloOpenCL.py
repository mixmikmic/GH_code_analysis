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

get_ipython().run_cell_magic('cl_kernel', '', '__kernel void add_kernel(__global const float *a, __global const float *b, __global float *c) {\n  int gid = get_global_id(0);\n  c[gid] = a[gid] + b[gid];\n}')

def opencl_add(a, b):
    #Make sure that the data is single precision floating point
    assert(np.issubdtype(a.dtype, np.float32))
    assert(np.issubdtype(b.dtype, np.float32))

    #Check that they have the same length
    assert(a.shape == b.shape)

    #Upload data to the device
    mf = cl.mem_flags
    a_g = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

    #Allocate output data
    c_g = cl.Buffer(cl_ctx, mf.WRITE_ONLY, a.nbytes)

    #Execute program on device
    add_kernel(cl_queue, a.shape, None, a_g, b_g, c_g)

    #Allocate data on the host for result
    c = np.empty_like(a)

    #Copy data from device to host
    cl.enqueue_copy(cl_queue, c, c_g)

    #Return result
    return c

#Create test input data
a = np.random.rand(50000).astype(np.float32)
b = np.random.rand(50000).astype(np.float32)

#Add using OpenCL
c = opencl_add(a, b)

#Compute reference using Numpy
c_ref = a + b

#Print result
print("C   = ", c)
print("Ref = ", c_ref)
print("Sad = ", np.sum(np.abs(c - c_ref)))




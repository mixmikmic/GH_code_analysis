import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import numba
from numba import jit
import sys
import os
import mkl
get_ipython().run_line_magic('matplotlib', 'inline')

def mandelbrot_image(fun, xmin, xmax, ymin, ymax, width=7, height=7, maxiter=80, cmap='magma'):
    dpi = 72
    img_width = dpi * width
    img_height = dpi * height
    z = fun(xmin,xmax,ymin,ymax,img_width,img_height,maxiter)
    
    fig, ax = plt.subplots(figsize=(width, height),dpi=72)
    plt.xticks([])
    plt.yticks([])
    plt.title("[{xmin}, {ymin}] to [{xmax}, {ymax}]".format(**locals()))
    
    norm = colors.PowerNorm(0.3)
    ax.imshow(z,cmap=cmap,origin='lower',norm=norm)

def linspace(start, stop, n):
    step = float(stop - start) / (n - 1)
    return [start + i * step for i in range(n)]

def mandel1(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return n

def mandel_set1(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80):
    r = linspace(xmin, xmax, width)
    i = linspace(ymin, ymax, height)
    n = [[0]*width for _ in range(height)]
    for x in range(width):
        for y in range(height):
            n[y][x] = mandel1(complex(r[x], i[y]), maxiter)
    return n

get_ipython().run_line_magic('timeit', 'mandel_set1()')

get_ipython().run_line_magic('timeit', 'mandel_set1(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

mandelbrot_image(mandel_set1, -2.0, 0.5, -1.25, 1.25)

mandelbrot_image(mandel_set1, -0.74877,-0.74872,0.06505,0.06510,maxiter=2048)

p = get_ipython().run_line_magic('prun', '-r -q mandel_set1()')
p.stream = sys.stdout
p.sort_stats('cumulative').print_stats(5)

get_ipython().run_line_magic('load_ext', 'line_profiler')

p = get_ipython().run_line_magic('lprun', '-r -f mandel1 mandel_set1()')
p.print_stats(sys.stdout)

@jit
def mandel2(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return n

def mandel_set2(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80):
    r = linspace(xmin, xmax, width)
    i = linspace(ymin, ymax, height)
    n = [[0]*width for _ in range(height)]
    for x in range(width):
        for y in range(height):
            n[y][x] = mandel2(complex(r[x], i[y]), maxiter)
    return n

get_ipython().run_line_magic('timeit', 'mandel_set2()')

get_ipython().run_line_magic('timeit', 'mandel_set2(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

mandelbrot_image(mandel_set2, -2.0, 0.5, -1.25, 1.25)

p = get_ipython().run_line_magic('prun', '-r -q mandel_set2()')
p.stream = sys.stdout
p.sort_stats('cumulative').print_stats()

p = get_ipython().run_line_magic('lprun', '-f mandel_set2 -r mandel_set2()')
p.print_stats(sys.stdout)

@jit
def mandel3(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return 0

def mandel_set3(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80):
    r = np.linspace(xmin, xmax, width)
    i = np.linspace(ymin, ymax, height)
    n = np.empty((height, width), dtype=int)
    for x in range(width):
        for y in range(height):
            n[y, x] = mandel3(complex(r[x], i[y]), maxiter)
    return n

get_ipython().run_line_magic('timeit', 'mandel_set3()')

get_ipython().run_line_magic('timeit', 'mandel_set3(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

@jit
def mandel4(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return n

@jit
def mandel_set4(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80):
    r = np.linspace(xmin, xmax, width)
    i = np.linspace(ymin, ymax, height)
    n = np.empty((height, width), dtype=int)
    for x in range(width):
        for y in range(height):
            n[y, x] = mandel4(complex(r[x], i[y]), maxiter)
    return n

get_ipython().run_line_magic('timeit', 'mandel_set4()')

get_ipython().run_line_magic('timeit', 'mandel_set4(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

@jit
def mandel5(creal, cimag, maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2 * real*imag + cimag
        real = real2 - imag2 + creal       
    return n

@jit
def mandel_set5(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80):
    r = np.linspace(xmin, xmax, width)
    i = np.linspace(ymin, ymax, height)
    n = np.empty((height, width), dtype=int)
    for x in range(width):
        for y in range(height):
            n[y, x] = mandel5(r[x], i[y], maxiter)
    return n

get_ipython().run_line_magic('timeit', 'mandel_set5()')

get_ipython().run_line_magic('timeit', 'mandel_set5(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

mandelbrot_image(mandel_set5, -2.0, 0.5, -1.25, 1.25)

mandelbrot_image(mandel_set5, -0.74877,-0.74872,0.06505,0.06510,maxiter=2048)

get_ipython().run_line_magic('load_ext', 'cython')

get_ipython().run_cell_magic('cython', '', 'import cython\nimport numpy as np\n\ncdef int mandel6(double creal, double cimag, int maxiter):\n    cdef:\n        double real2, imag2\n        double real = creal, imag = cimag\n        int n\n\n    for n in range(maxiter):\n        real2 = real*real\n        imag2 = imag*imag\n        if real2 + imag2 > 4.0:\n            return n\n        imag = 2* real*imag + cimag\n        real = real2 - imag2 + creal;\n    return n\n\n@cython.boundscheck(False) \n@cython.wraparound(False)\ncpdef mandel_set6(double xmin, double xmax, double ymin, double ymax, int width, int height, int maxiter):\n    cdef:\n        double[:] r1 = np.linspace(xmin, xmax, width)\n        double[:] r2 = np.linspace(ymin, ymax, height)\n        int[:,:] n = np.empty((height, width), np.int32)\n        int i,j\n    \n    for i in range(width):\n        for j in range(height):\n            n[j,i] = mandel6(r1[i], r2[j], maxiter)\n    return n')

get_ipython().run_line_magic('timeit', 'mandel_set6(-2, 0.5, -1.25, 1.25, 1000, 1000, 80)')

get_ipython().run_line_magic('timeit', 'mandel_set6(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

mandelbrot_image(mandel_set6, -0.74877,-0.74872,0.06505,0.06510,maxiter=2048)

import pyopencl as cl
platforms = cl.get_platforms()
print(platforms)
ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])

def mandel7(q, maxiter):

    global ctx
    
    queue = cl.CommandQueue(ctx)
    
    output = np.empty(q.shape, dtype=np.uint16)

    prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float real = q[gid].x;
        float imag = q[gid].y;
        output[gid] = maxiter;
        for(int curiter = 0; curiter < maxiter; curiter++) {
            float real2 = real*real, imag2 = imag*imag;
            if (real*real + imag*imag > 4.0f){
                 output[gid] = curiter;
                 return;
            }
            imag = 2* real*imag + q[gid].y;
            real = real2 - imag2 + q[gid].x;
            
        }
    }
    """).build()

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)


    prg.mandelbrot(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()
    
    return output

def mandel_set7(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    c = np.ravel(c)
    n = mandel7(c,maxiter)
    n = n.reshape((height, width))
    return n

get_ipython().run_line_magic('timeit', 'mandel_set7(-2, 0.5, -1.25, 1.25, 1000, 1000, 80)')

get_ipython().run_line_magic('timeit', 'mandel_set7(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

mandelbrot_image(mandel_set7, -0.74877,-0.74872,0.06505,0.06510,maxiter=2048)

get_ipython().run_line_magic('cat', 'mandel08.f90')

from mb_fort import mandel8, mandel_set8

get_ipython().run_line_magic('timeit', 'mandel_set8(-2, 0.5, -1.25, 1.25, 1000, 1000, 80)')

get_ipython().run_line_magic('timeit', 'mandel_set8(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

mandelbrot_image(mandel_set8, -0.74877,-0.74872,0.06505,0.06510, maxiter=2048)

get_ipython().run_cell_magic('writefile', 'mandel09.py', "from numba import jit, vectorize, complex64, int32\nimport numpy as np\n\n@vectorize([int32(complex64, int32)], target='parallel')\ndef mandel9(c, maxiter):\n    nreal = 0\n    real = 0\n    imag = 0\n    for n in range(maxiter):\n        nreal = real*real - imag*imag + c.real\n        imag = 2* real*imag + c.imag\n        real = nreal;\n        if real * real + imag * imag > 4.0:\n            return n\n    return n\n        \ndef mandel_set9(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80):\n    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)\n    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)\n    c = r1 + r2[:,None]*1j\n    n = mandel9(c,maxiter)\n    return n")

get_ipython().run_cell_magic('bash', '', 'expr=\'mandel09.mandel_set9(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)\'\nfor threads in 1 2 4 8 14; do\n    printf "%3i threads\\n" ${threads}\n    NUMBA_NUM_THREADS=${threads} python -m timeit -s \'import mandel09\' "$expr"\ndone')

import multiprocessing as mp

ncpus = 1
@jit
def mandel10(creal, cimag, maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2 * real*imag + cimag
        real = real2 - imag2 + creal       
    return n

@jit
def mandel10_row(args):
    y, xmin, xmax, width, maxiter = args
    r = np.linspace(xmin, xmax, width)
    res = [0] * width
    for x in range(width):
        res[x] = mandel10(r[x], y, maxiter)
    return res
        

def mandel_set10(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80):
    i = np.linspace(ymin, ymax, height)
    with mp.Pool(ncpus) as pool:
        n = pool.map(mandel10_row, ((a, xmin, xmax, width, maxiter) for a in i))
    return n

for i in (1, 2, 4, 8, 14):
    ncpus = i
    get_ipython().run_line_magic('timeit', 'mandel_set10(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)')

mandelbrot_image(mandel_set10, -2.0, 0.5, -1.25, 1.25)

mandelbrot_image(mandel_set10, -0.74877,-0.74872,0.06505,0.06510, maxiter=2048)


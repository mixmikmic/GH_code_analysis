import math
import numpy

import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule

# compute a MILLION values
PROBLEM_SIZE = int(1e6)

# generate a CUDA (C-ish) function that will run on the GPU; PROBLEM_SIZE is hard-wired
module = SourceModule("""
__global__ void just_multiply(float *dest, float *a, float *b)
{
  // function is called for ONE item; find out which one
  const int id = threadIdx.x + blockDim.x*blockIdx.x;
  if (id < %d)
    dest[id] = a[id] * b[id];
}
""" % PROBLEM_SIZE)

# pull "just_multiply" out as a Python callable
just_multiply = module.get_function("just_multiply")

# create Numpy arrays on the CPU
a = numpy.random.randn(PROBLEM_SIZE).astype(numpy.float32)
b = numpy.random.randn(PROBLEM_SIZE).astype(numpy.float32)
dest = numpy.zeros_like(a)

# define block/grid size for our problem: at least 512 threads at a time (might do more)
# and we're only going to use x indexes (the y and z sizes are 1)
blockdim = (512, 1, 1)
griddim = (int(math.ceil(PROBLEM_SIZE / 512.0)), 1, 1)

# copy the "driver.In" arrays to the GPU, run the 
just_multiply(driver.Out(dest), driver.In(a), driver.In(b), block=blockdim, grid=griddim)

# compare the GPU calculation (dest) with a CPU calculation (a*b)
print dest - a*b

module2 = SourceModule("""
__global__ void mapper(float *dest)
{
  const int id = threadIdx.x + blockDim.x*blockIdx.x;
  const double x = 1.0 * id / %d;     // x goes from 0.0 to 1.0 in PROBLEM_SIZE steps
  if (id < %d)
    dest[id] = 4.0 / (1.0 + x*x);
}
""" % (PROBLEM_SIZE, PROBLEM_SIZE))

mapper = module2.get_function("mapper")
dest = numpy.empty(PROBLEM_SIZE, dtype=numpy.float32)
blockdim = (512, 1, 1)
griddim = (int(math.ceil(PROBLEM_SIZE / 512.0)), 1, 1)

mapper(driver.Out(dest), block=blockdim, grid=griddim)

dest.sum() * (1.0 / PROBLEM_SIZE)  # correct for bin size

module3 = SourceModule("""
__global__ void reducer(float *dest, int i)
{
  const int PROBLEM_SIZE = %d;
  const int id = threadIdx.x + blockDim.x*blockIdx.x;
  if (id %% (2*i) == 0  &&  id + i < PROBLEM_SIZE) {
    dest[id] += dest[id + i];
  }
}
""" % PROBLEM_SIZE)

blockdim = (512, 1, 1)
griddim = (int(math.ceil(PROBLEM_SIZE / 512.0)), 1, 1)

reducer = module3.get_function("reducer")

# Python for loop over the 20 steps to reduce the array
i = 1
while i < PROBLEM_SIZE:
    reducer(driver.InOut(dest), numpy.int32(i), block=blockdim, grid=griddim)
    i *= 2

# final result is in the first element
dest[0] * (1.0 / PROBLEM_SIZE)

# allocate the array directly on the GPU, no CPU involved
dest_gpu = driver.mem_alloc(PROBLEM_SIZE * numpy.dtype(numpy.float32).itemsize)

# do it again without "driver.InOut", which copies Numpy (CPU) to and from the GPU
mapper(dest_gpu, block=blockdim, grid=griddim)
i = 1
while i < PROBLEM_SIZE:
    reducer(dest_gpu, numpy.int32(i), block=blockdim, grid=griddim)
    i *= 2

# we only need the first element, so create a Numpy array with exactly one element
only_one_element = numpy.empty(1, dtype=numpy.float32)

# copy just that one element
driver.memcpy_dtoh(only_one_element, dest_gpu)

print only_one_element[0] * (1.0 / PROBLEM_SIZE)


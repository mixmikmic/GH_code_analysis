from __future__ import division
from numba import cuda, float32
import numpy
import math
import time

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

n = 2**11  # we'll use square matrices
n

A = numpy.full((n, n), 1, numpy.float32)  # both unit matrices
B = numpy.full((n, n), 1, numpy.float32) 

get_ipython().run_line_magic('timeit', 'A @ B  # raw numpy')
A @ B

# naive kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

# Copy the arrays to the device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

# Allocate memory on the device for the result
C_global_mem = cuda.device_array(A.shape)

# Configure the blocks
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Start the kernel 
t1 = time.clock()
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
t2 = time.clock()

# Copy the result back to the host
C = C_global_mem.copy_to_host()

print('GPU time - cold (ms): {0:,.3f}'.format(1000*(t2 - t1)))
print(C)

C_global_mem = cuda.device_array(A.shape)
t1 = time.clock()
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
t2 = time.clock()

print('\nGPU time - warm (ms): {0:,.3f}'.format(1000*(t2 - t1)))

res = C_global_mem.copy_to_host()
print(C)

# revised kernel with more efficient sharing of memory
@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)
C_global_mem = cuda.device_array(A.shape) 

# Configure the blocks
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Start the kernel 
t1 = time.clock()
fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
t2 = time.clock()

res = C_global_mem.copy_to_host()

print('GPU time - cold (ms): {0:,.3f}'.format(1000*(t2 - t1)))
print(C)

C_global_mem = cuda.device_array(A.shape)
t1 = time.clock()
fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
t2 = time.clock()

print('\nGPU time - warm (ms): {0:,.3f}'.format(1000*(t2 - t1)))

res = C_global_mem.copy_to_host()
print(C)


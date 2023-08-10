import numpy as np
import torch
import time

n = 10000
A = np.random.randn(n, n)

start = time.time()
A2 = np.matmul(A, A)
print('%.4f seconds' % (time.time() - start))

A = torch.from_numpy(A).cuda() # to cast a tensor on a GPU we only have to cast its type to `cuda`
start = time.time()
A2 = torch.mm(A, A)
print('%.4f seconds' % (time.time() - start))

print('Number of GPUs: %i' % torch.cuda.device_count())
print('ID of the GPU used: %i' % torch.cuda.current_device()) # current default GPU
torch.cuda.set_device(1) # switch to GPU 1
print('ID of the GPU used: %i' % torch.cuda.current_device())

# Using context manager to place operations on a given device
with torch.cuda.device(0):
    A = torch.randn(n, n).cuda()
    A2 = A.mm(A)
print('A is on GPU %i' % (A.get_device()))
      
with torch.cuda.device(3):
    A = torch.randn(n, n).cuda()
    A2 = A.mm(A)
print('A is on GPU %i' % (A.get_device()))


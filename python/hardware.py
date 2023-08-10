# for linux, to see CPU info
get_ipython().system('cat /proc/cpuinfo')
# for OSX, it is
# !sysctl -a|grep machdep.cpu 

operations_per_cycle = 256 / 32  # single precision (4 byte)
GFLOPS_CPU = 2 * 2.5 * operations_per_cycle
print('Theoretical computation power of my CPU = %s GFLOPS'%GFLOPS_CPU)

get_ipython().system('nvidia-settings')

GFLOPS_CPU = 384 * 2 * 1.189
print('Theoretical computation power of my GPU = %s GFLOPS'%GFLOPS_CPU)

import torch

N = 1024
A = torch.randn(N, N)
B = torch.randn(N, N)
get_ipython().run_line_magic('timeit', '-n 100 C = A.mm(B)')

# upload data to GPU, do the same calculation
AC = A.cuda()
BC = B.cuda()
get_ipython().run_line_magic('timeit', '-n 100 CC = AC.mm(BC)')

# run only if you have an Nvidia graphic card, with driver properly configured
get_ipython().system('nvidia-smi')




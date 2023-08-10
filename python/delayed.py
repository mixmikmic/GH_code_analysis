from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4, n_workers=1)
client

import time
import random

def inc(x):
    time.sleep(random.random())
    return x + 1

def dec(x):
    time.sleep(random.random())
    return x - 1
    
def add(x, y):
    time.sleep(random.random())
    return x + y 

get_ipython().run_cell_magic('time', '', 'x = inc(1)\ny = dec(2)\nz = add(x, y)\nz')

import dask
inc = dask.delayed(inc)
dec = dask.delayed(dec)
add = dask.delayed(add)

get_ipython().run_cell_magic('time', '', 'x = inc(1)\ny = dec(2)\nz = add(x, y)\nz')

z.visualize(rankdir='LR')

z.compute()

get_ipython().run_cell_magic('time', '', 'zs = []\nfor i in range(256):\n    x = inc(i)\n    y = dec(x)\n    z = add(x, y)\n    zs.append(z)\n    \nzs = dask.persist(*zs)  # trigger computation in the background')

for i in range(10):
    client.cluster.start_worker(ncores=4)

L = zs
while len(L) > 1:
    new_L = []
    for i in range(0, len(L), 2):
        lazy = add(L[i], L[i + 1])  # add neighbors
        new_L.append(lazy)
    L = new_L                       # swap old list for new

dask.compute(L)




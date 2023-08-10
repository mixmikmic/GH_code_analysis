from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4, n_workers=1)
client

import time
import random

def inc(x):
    time.sleep(random.random())
    return x + 1

def double(x):
    time.sleep(random.random())
    return 2 * x
    
def add(x, y):
    time.sleep(random.random())
    return x + y 

inc(1)

future = client.submit(inc, 1)  # returns immediately with pending future
future

future  # scheduler and client talk constantly

future.result()

x = client.submit(inc, 1)
y = client.submit(double, 2)
z = client.submit(add, x, y)
z

z.result()

get_ipython().run_cell_magic('time', '', 'zs = []\nfor i in range(256):\n    x = client.submit(inc, i)     # x = inc(i)\n    y = client.submit(double, x)  # y = inc(x)\n    z = client.submit(add, x, y)  # z = inc(y)\n    zs.append(z)\n    \ntotal = client.submit(sum, zs)')

for i in range(10):
    client.cluster.start_worker(ncores=4)

L = zs
while len(L) > 1:
    new_L = []
    for i in range(0, len(L), 2):
        future = client.submit(add, L[i], L[i + 1])  # add neighbors
        new_L.append(future)
    L = new_L                                   # swap old list for new
   

del future, L, new_L, total  # clear out some old work

from dask.distributed import as_completed

zs = client.map(inc, zs)
seq = as_completed(zs)

while seq.count() > 2:  # at least two futures left
    a = next(seq)
    b = next(seq)
    new = client.submit(add, a, b)  # add them together
    seq.add(new)                    # add new future back into loop




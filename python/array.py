from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4, n_workers=1, memory_limit='2GB')
client

import dask.array as da
x = da.random.random((10000, 10000), chunks=(1000, 1000))
x

y = x + x.T
z = y[::2, 5000:].mean(axis=1)
z

z.compute()

y = y.persist()

get_ipython().run_line_magic('time', 'y[0, 0].compute()')

get_ipython().run_line_magic('time', 'y.sum().compute()')


import numpy as np
import dask.array as da

A = np.random.rand(2000,4000)

get_ipython().magic('whos')

A

get_ipython().magic('time B = A**2 + np.sin(A) * A * np.log(A)')

A_dask = da.from_array(A, chunks=(1000, 1000))

A_dask.numblocks

get_ipython().magic('time B_dask = (A_dask**2 + da.sin(A_dask) * A_dask * da.log(A_dask)).compute(num_workers=1)')

get_ipython().magic('time B_dask = (A_dask**2 + da.sin(A_dask) * A_dask * da.log(A_dask)).compute(num_workers=4)')

assert np.allclose(B, B_dask)




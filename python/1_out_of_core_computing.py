get_ipython().system('ls -lah ../data/random.hdf5')

import h5py
import os
f = h5py.File(os.path.join('..', 'data', 'random.hdf5'))
dset = f['/x']

dset.shape[0] / 1e6

dset.dtype

import dask.array as da
x = da.from_array(dset, chunks=(int(1e6),))

result = x[:int(4e7)].mean()
result

get_ipython().magic('time result.compute(num_workers=4)')




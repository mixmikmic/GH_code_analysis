import numpy as np

# create an array of normally-distributed random numbers
a = np.random.normal(size=1000)

# multiply this array by a factor
b = a * 4

# find the minimum value
b_min = b.min()
print(b_min)

import dask.array as da

a = da.random.normal(size=1000, chunks=200)

b = a * 4

b_min = b.min()
print(b_min)

b_min.compute()

b_min.visualize()






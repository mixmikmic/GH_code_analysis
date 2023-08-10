import numpy

# Python Convention
import numpy as np

# Generate random data
# np ==> short for Numpy
# random ==> sub-module in Numpy
# randn ==> a function in sub-module, random, in Numpy
arr1 = np.random.randn(2, 3) # ==> 2 rows, 3 columns

arr1

# basic attributes on the array
arr1.ndim, arr1.shape, arr1.dtype

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2

# we can a specify data type
arr3 = np.array(data2, dtype=np.float64)
arr3

# or cast an array using astype method
arr4 = arr3.astype('int64')
arr3.dtype, arr4.dtype

# element-wise calculations
arr3 * 2

# array-wise calculations
arr3 - arr3

# create an array using arange, similar to python's built-in range
arr = np.arange(10) # again, 10 elements from 0 to 9
arr

# retrieve element(s)
arr[5], arr[5:]

# update element(s).
arr[5:] = -99
arr

# WARNING: mutations, need to use .copy() method
arr_slice = arr[5:]
arr_slice[1]=-100
arr

# generate random data
data = np.random.randn(2,5)
data

# create a new array which is booleans
cond = data <= 0
cond

# filter data using conditions
data_cond = data[cond]
data_cond

# create an array
arr = np.arange(5)
arr

# universal (element-wise) functions: abs, square, exp, log, and so on
arr_sqrt = np.sqrt(arr) # <= fast element-wise operations
arr_sqrt

x = np.random.randn(5)
x

y = np.random.randn(5)
y

# binary (array-wise) function
# obtain the maxium values between two arrays
np.maximum(x, y)

arr = np.random.randn(2, 5) # <= 2 rows, 5 columns

# obtain the mean of elements in the array
arr.mean(), np.mean(arr)

# what if we want row-wise mean instead of whole array?
arr.mean(axis=1), np.mean(arr, axis=1)

# what if column-wise?
arr.mean(axis=0), np.mean(arr, axis=0)

# Python built-in "loop" style
get_ipython().run_line_magic('matplotlib', 'inline')
import random, matplotlib.pyplot as plt
position = 0 # <== starting point
walk = [position] # <== a list with the starting point
steps = 100 # <= 100 steps
for i in range(steps):
    step = 1 if random.randint(0,1) else -1 # 0 is False, 1 is True
    position += step # <== incremental operations: position = position + step
    walk.append(position) # append new position to the list
# plot data
plt.plot(walk);

nsteps = 1000

draws = np.random.randint(0, 2, size=nsteps) # <= random draw from 0, 1

np_steps = np.where(draws > 0, 1, -1) # if draw = 1 then step 1, otherwise, -1

np_walk = np.cumsum(np_steps) # or np_steps.cumsum(), NumPy method, cumulative sum

plt.plot(np_walk);

nwalks = 10000 # 10,000 simulations
nsteps = 1000

draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # <== again 0 or 1
steps = np.where(draws > 0, 1, -1) # change 0 to -1
walks = steps.cumsum(axis=1) # apply the cumsum() method across columns...

walks # 10000 simulations... all in an array....


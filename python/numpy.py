import numpy as np
from numpy.random import randn

arr = np.array([[0.1,1,3],[1,2,3]])
print arr
print arr.shape
print arr.dtype

arr2 = np.zeros([4])
print arr2
print arr2.shape
print arr2.dtype

arr * arr

arr - arr

print 1 / arr

arr ** 0.5

arr_slice = arr[1:]
arr_slice[:] = 7,8,9
print arr

arr[0, 1]  # the index is actually a tuple

arr[0:, :-1] #the index is actually a Slice type

data = randn(6, 4)
names = np.array(['a', 'b', 'c', 'a', 'b', 'a'])
print data

names == 'a'

data[names == 'a']

mask = (names == 'c') | (names == 'b')
data[mask, :3]

data < 0

data[data<0] = 0
print data

arr = np.empty((8,4))
for i in range(8): arr[i] = i
    
print arr

arr[[1,3,0]]

arr[[4,5,6], [1,2,3]] # will return elements (4,1), (5,2) and (1,3) which is not trivial

arr[np.ix_([4,5,6], [1,2])] # now the len of the lists don't have to be equal.

arr = np.arange(15).reshape(3,5)
arr

arr.T

np.dot(arr.T, arr)

arr.swapaxes(0,1)

np.sqrt([81 , 64, 4])

np.maximum([-1, 8, 0.3], [1, 7, 10.1])

xarr = np.array([1,2,3])
yarr = np.array([4,5,6])
print np.where(xarr < 2 , xarr, yarr)

arr = np.random.rand(2, 2)
arr

print arr.mean()
print arr.mean(axis=0)
print arr.sum()
print arr.sum(axis=1) # axis.. 0 for column, 1 for row

arr = randn(100)
bools = arr>0

bools.all()

bools.any()

bools.sum()

arr = randn(10)

arr.sort()
print arr

x = np.array(['x', 'y', 'z', 'x'])
np.unique(x)




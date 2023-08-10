get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import numpy.random as npr
vsep = "\n-------------------\n"

def dump_array(arr):
    print("%s array of %s:" % (arr.shape, arr.dtype))
    print(arr)

arr1 = np.arange(5)
print("arr1:\n", arr1, end=vsep)

print("arr1 + scalar:\n", arr1+10, end=vsep)

print("arr1 + arr1 (same shape):\n", arr1+arr1, end=vsep)

arr2 = np.arange(5).reshape(5,1) * 10
arr3 = np.arange(5).reshape(1,5) * 100
print("arr2:\n", arr2)
print("arr3:\n", arr3, end=vsep)

print("arr1 + arr2 [ %s + %s --> %s ]:" % 
      (arr1.shape, arr2.shape, (arr1 + arr2).shape))
print(arr1+arr2, end=vsep)
print("arr1 + arr3 [ %s + %s --> %s ]:" % 
      (arr1.shape, arr3.shape, (arr1 + arr3).shape))
print(arr1+arr3)

arr1 = np.arange(6).reshape(3,2)
arr2 = np.arange(10, 40, 10).reshape(3,1)

print("arr1:")
dump_array(arr1)
print("\narr2:")
dump_array(arr2)
print("\narr1 + arr2:")
print(arr1+arr2)

a1 = np.array([1,2,3])       # 3 -> 1x3
b1 = np.array([[10, 20, 30], # 2x3
               [40, 50, 60]]) 
print(a1+b1)

result = (np.ones((  6,1)) +  # 3rd dimension replicated
          np.ones((1,6,4)))
print(result.shape)

result = (np.ones((3,6,1)) + 
          np.ones((1,6,4)))   # 1st and 3rd dimension replicated
print(result.shape)

arr1 = np.arange(6).reshape((2,3))  # 2x3
arr2 = np.array([10, 100])          #   2
arr1 + arr2  # This will fail

# let's massage the shape
arr3 = arr2[:, np.newaxis] # arr2 -> 2x1
print("arr3 shape:", arr3.shape)
print("arr1 + arr3")
print(arr1+arr3)

arr = np.array([10, 100])
print("original shape:", arr.shape)

arrNew = arr2[np.newaxis, :]
print("arrNew shape:", arrNew.shape)

arr1 = np.arange(0,6).reshape(2,3)
arr2 = np.arange(10,22).reshape(4,3)
np.tile(arr1, (2,1)) * arr2




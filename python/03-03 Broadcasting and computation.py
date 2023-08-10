from __future__ import print_function
import numpy as np
arr1 = np.random.randint(1, 40, 5)
num  = 5
print("Arr1: \n{}".format(arr1), end="\n\n")
print("num : \n{}".format(num), end="\n\n")
print("Sum : \n{}".format(arr1+num), end="\n\n")

arr1 = np.ones((3, 4))
arr2 = np.arange(4)

arr1 + arr2

print("Arr1: \n{}".format(arr1))
print("arr1.shape: {}".format(arr1.shape), end="\n\n")
print("Arr1 Transpose: \n{}".format(arr1.T))
print("arr1.T.shape: {}".format(arr1.T.shape))

arr1 + arr1.T

arr = np.random.randint(1, 700, 10000)
print("Sum of 1D: {}".format(np.sum(arr)))

arr2d = np.random.uniform(1, 700, (3, 4))
print("Sum of 2D: {}".format(np.sum(arr2d)))

print("Max of 1D arr: {}".format(np.max(arr)))
print("Min of 1D arr: {}".format(np.min(arr)))
print("Max of 2D arr: {}".format(np.max(arr2d)))
print("Min of 2D arr: {}".format(np.min(arr2d)))

print("2D Array: \n{}".format(arr2d), end="\n\n")
print("Max of 2D arr along axis 0: {}".format(np.amax(arr2d, axis=0)))
print("Min of 2D arr along axis 0: {}".format(np.amin(arr2d, axis=0)))

print("2D Array: \n{}".format(arr2d), end="\n\n")
print("Std along axis 0: \n{}".format(np.std(arr2d, axis=0)))
print("Std along axis 1: \n{}".format(np.std(arr2d, axis=1)))


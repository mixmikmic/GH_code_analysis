from __future__ import print_function
import numpy as np
# Single dimensional Array from a list
arr = np.array([1, 2, 3, 4], dtype=float)
print('Type: ',type(arr))
print('Shape: ',arr.shape)
print('Dimension: ',arr.ndim)
print('Itemsize: ',arr.itemsize)
print('Size: ',arr.size)

# Elements have to be of same datatype
arr = np.array([1, 2.0, "ucsl"])
print("Datatype: ", arr.dtype)

arr1 = np.arange(5, dtype=float)
print('arange() with float dtype: \n',arr1)
# Divide the range between start and stop in equal `num` intervals
arr2 = np.linspace(0, 8, num=5)
print('\n linspace(): \n', arr2)
arr3 = np.ones((2, 3), dtype=float)
print ('\n ones(): \n',arr3)
arr4 = np.zeros((2,3), dtype=float)
print ('\n zeros(): \n',arr4)
arr5 = np.empty((2, 4))
print('\n Empty: \n',arr5)  # Your output may be different..
arr6 = np.ones_like(arr1)
print('\n Ones_like(): \n',arr6)
arr7 = np.diag(arr1)
print('\n Diagonal array: \n',arr7)

arr2d = np.arange(27).reshape(3, 9)
print("2D array: \n{}\n".format(arr2d))
arr3d = np.arange(27).reshape(3,3,3)
print("3D array: \n{}\n".format(arr3d))

arr = np.arange(3, 10)
print(arr[4])

print(arr[-3])

arr3d = np.arange(27).reshape(3,3,3)
dim, row, col = 2, 1, 0
print("3D array: \n", arr3d, end="\n\n")
print("Element at {dim}, {row}, {col} is: {val}".format(dim=dim, 
                                                        row=row, 
                                                        col=col, 
                                                        val=arr3d[dim, row, col]))

arr = np.linspace(0, 8, num=5)
print("Original Array: \n", arr, end="\n\n")
# let the slicings begin
print("arr[:3]: ", arr[:3])
print("arr[-5:5:2]: ", arr[-5:5:2])
print("arr[::2]: ", arr[::2])
# Reverse the elements
print("arr[::-1]: ", arr[::-1])
# Reverse every other array from index 2
print("arr[2::-2]: ", arr[2::-2])

# Array of random integers between low and high of fixed size(mxn)
arr = np.random.randint(low=0, high=100, size=(3,4))
print("2D array: \n", arr, end="\n\n")
# first row, three columns
print("first row, three columns: \n", arr[:1, :3], end="\n\n")
# all rows, third column
print("all rows, third column: \n", arr[:, 3], end="\n\n")
# changing dimensions 
print("reversing rows and columns together: \n",
     arr[::-1, ::-1], end="\n\n")

arr1 = np.arange(5)
# slice arr1
arr2 = arr1[3:5]
print("arr1: \n", arr1, end="\n\n")
print("Sliced array: \n", arr2)
print('\nBefore changing, arr2[0]: \n',arr2[0])
# change value for 0th element of the slice
arr2[0] = 99
print('\nAfter changing arr2[0], arr1: \n',arr1)

arr = np.random.randint(low=0, high=100, size=12)
print("Original Array: \n", arr, end="\n\n")
print("Reshaped to 3 x 4: \n", arr.reshape(3,4), end="\n\n")
print("Row vector : \n", arr[np.newaxis, :], end="\n\n")
print("Column vector : \n", arr[:, np.newaxis], end="\n\n")

# Creating two 1D arrays separately
arr1 = np.arange(10)
arr2 = np.arange(10, 20)
arr3 = np.concatenate((arr1, arr2))
print("Arr1: \n{}".format(arr1), end="\n\n")
print("Arr2: \n{}".format(arr2), end="\n\n")
print("Concatenated Array: \n{}".format(arr3))

arr1 = np.random.randint(1, 10, 8).reshape(2, 4)
arr2 = np.random.randint(90, 100, 8).reshape(2, 4)
# stacking horizontally
hs_arr = np.hstack((arr1, arr2))
# stacking vertically
vs_arr = np.vstack((arr1, arr2))
print("Arr1: \n{}".format(arr1), end="\n\n")
print("Arr2: \n{}".format(arr2), end="\n\n")
print("Horizontally Stacked Array: \n{}".format(hs_arr), end="\n\n")
print("Vertically Stacked Array: \n{}".format(vs_arr), end="\n\n")

arr1 = np.arange(20)
np.split(arr1, (2, 8, 10, 14))

arr2d = np.random.randint(0, 9, (3,3))
print("Original Array: \n{}".format(arr2d), end="\n\n")
# split along horizontal axis
arr1, arr2 = np.hsplit(arr2d, [2])
print("First Split: \n{}".format(arr1), end="\n\n")
print("Remaining Split: \n{}".format(arr2), end="\n\n")


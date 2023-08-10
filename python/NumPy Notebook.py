my_list =[1,2,3]

my_list

import numpy as np

arr= np.array(my_list)

arr

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]

np.array(my_matrix)

np.arange(0,10)

#arange was used to create a 1d array in a specific range

np.arange(0,11,2)

#arrays of all zeroes

np.zeros(3)

np.zeros((5,5)) # create 2 d array

#array of ones

np.ones(4)

np.ones((3,4))

#linspace - it takes no of points as well

np.linspace(0,5,10) # 10 is no of evenly spaced pointss

np.eye(4) # identity matrix of 4*4

np.random.rand(5) # 1-d array of size 5 between 0 and 1 by default

np.random.rand(5,5) # 2-d array

np.random.randn(4,4) # normalized distrbution with randn

np.random.randint(1,100,6) # random nos array between low and high range

arr = np.arange(25)

arr

ranarr = np.random.randint(0,50,10)

#reshape method common method

arr.reshape(5,5) #converted 1-d to 2-d array

ranarr

ranarr.max()

ranarr.min()

ranarr

ranarr.argmax()

arr.shape

arr= arr.reshape(5,5)

arr.shape

#datatype of an array

arr.dtype

from numpy.random import randint

randint(2,10)

#indexing and selection of numpy arrays

#grouping elements

arr= np.arange(0,11)

arr

arr[8]

arr[0:7]

arr[0:5]

arr[:6]

arr[5:]

#broadcast

arr[0:5] = 100

arr

arr = np.arange(0,11)

arr

slice_of_arr = arr[0:6]

slice_of_arr

slice_of_arr[:]=99

slice_of_arr

arr

#slice value changed and also changed original array arr, pass by value

#for copy and not original

arr_copy = arr.copy()

arr_copy

arr_copy[:]=100

arr_copy

arr

#2-d arrays

arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])

arr_2d

arr_2d[0][0] # getting element in 2d array

arr_2d[0]

arr_2d[2][1]

arr_2d[1,2] # comma notation

arr_2d[1][2]

arr_2d[:2,1:]

arr_2d[:2]



arr_2d[1:2,:]

#conditional selection

arr = np.arange(1,11)

arr

#getting boolean array out of this

bool_arr = arr>5

bool_arr

arr[bool_arr]

arr[arr>5]

arr[arr<3]

arr_2d = np.arange(50).reshape(5,10)

arr_2d

arr_2d[1:3,3:5]

#Numpy operations on arrays

#array with array , scalars and universal array functions

arr = np.arange(0,11)

arr

arr+ arr #addition

arr * arr

arr + 100 #scalar addition

arr/arr #gives nan where 0/0 occurs , got warning and no error

1/arr #got infinity, no error only warning

arr **2

#universal array functions

np.sqrt(arr)

np.exp(arr)

np.max(arr)

arr.max()

np.sin(arr)

np.log(arr)




import numpy as np  # numpy is often imported as np for short terminology


def print_numpy_attributes(a):
    """dumps ``numpy.ndarray`` attribute information 
    
    :param a: 
    :type a: np.ndarray 
    :return: 
    """
    print('----- numpy attributes info -----')
    print('', a)  # data will be printed. this is same with a.data
    print('type', type(a))  # should be <class 'numpy.ndarray'>
    print('data', a.data)   # actual array data
    print('dtype', a.dtype)  # type of data (int, float32 etc)
    print('shape', a.shape)  # dimensional information of data (2, 3) etc.
    print('ndim', a.ndim)  # total dimension of shape. 0 means scalar, 1 is vector, 2 is matrix... 
    print('size', a.size)  # total size of data, which is product sum of shape
    print('---------------------------------')

# 1. creating scalar
a1 = np.array(3)  # note that np.array([3]) will create vector with 1 element
print_numpy_attributes(a1)

# 2. creating vector
l2 = [1, 2, 3]
a2 = np.array(l2)
print(l2, type(l2))         # l2 is list
print_numpy_attributes(a2)  # a2 is numpy.ndarray

# 3. creating matrix
l3 = [[1, 2, 3], [4, 5, 6]]
a3 = np.array(l3)
# print(l3, type(l3))
print_numpy_attributes(a3)

# 4. creating general multi-dimensional array (tensor)
a4 = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[11, 22, 33], [44, 55, 66]]
    ]
)
print_numpy_attributes(a4)




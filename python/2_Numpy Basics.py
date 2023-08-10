# This is an import statement in Python, just like Java
import numpy as np 
#np is an alias for numpy in the above statement

# arange - returns a list of numbers from 0 to n-1
# reshape - reshapes the 1 x n number to the shape specified
a = np.arange(15).reshape(3, 5)
print (a)

# Find dimensions of an existing matrix
print (a.shape)

# Find the datatype of a variable
print (a.dtype)

# Casting a normal list to a numpy array
a = np.array([2,3,4])

# Create a matrix of shape 3 x 4 having all zeros
np.zeros((3,4))

# Create a 3-D matrix of shape 2 x 3 x 4 having all ones. 
# Specify the datatype to be 16-bit integer
np.ones((2,3,4), dtype=np.int16)                # dtype can also be specified

# Create an empty numpy array os shape 2 x 3
np.empty((2,3))

# arange takes 3 arguments here - start value, end value, step size
np.arange(10, 30, 5)

np.arange(5)

# Multiply all elements of the numpy array with a constant
a = np.ones((2,3), dtype=int)
a *= 3 # shorthand notation. Is equivalent to a = a*3
print (a)

# Generate random numbers in range 0 to 1 and create a numpy array of shape 2 x 3
np.random.seed(5) # Setting the seed ensures same random numbers each time with this seed
a = np.random.random((2,3))
print (a)


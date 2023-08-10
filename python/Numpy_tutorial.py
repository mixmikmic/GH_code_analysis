import numpy as np

a = np.array([1,2,3])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
print(a)

b = np.array([[1,2,3],[4,5,6]])
print(type(b))
print(b.shape)
# print(b[0,0], b[0,1], b[1,0])
print(b)

# create an array of all zeros
c = np.zeros((2,2))
print(c, end='\n------------------------------\n')

# create an array of all ones
d = np.ones((1,2))
print(d, end='\n------------------------------\n')

# create a constant array
e = np.full((2,2),7, np.int) #providing dtype to supress future warning
print(e, end='\n------------------------------\n')

# create a 3x3 identity matrix
f = np.eye(3,3)
print(f, end='\n------------------------------\n')

# create an array filled with random values
g = np.random.random((2,2))
print(g, end='\n------------------------------\n')

# create a 2-dimensional array with shape(3,4)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# use slicing to pull out sub-array consisting of first two rows
# and columns one and two
# b is an array of shape(2,2)
b = a[:2, 1:3]
print(b, end='\n------------------------------\n')

# a slice of an array is a view into the same data
# modifying it will modify the original array
print(a[0,1], end='\n------------------------------\n')
b[0,0] = 77
print(a[0,1], end='\n------------------------------\n')

# two ways of accessing the data in the middle row of the array.
# mixing integer indeixng with slices yields an array of lower rank.
# while using only slices yields an array of the same rank as the original array.

# rank 1 view of second row of array a
row_r1 = a[1, :]
# rank 2 view of second row of array a
row_r2 = a[1:2,:]

print(row_r1, row_r1.shape, '\n')
print(row_r2, row_r2.shape, '\n')

# making same distinction when accessing columns of an array
col_c1 = a[:, 1]
col_c2 = a[:, 1:2]

print(col_c1, col_c1.shape, '\n')
print(col_c2, col_c2.shape, '\n')

a = np.array([[1,2], [3,4], [5,6]])

print(a[[0, 1, 2], [0, 1, 0]])
# the above example of integer array indexing is similar to:
print(np.array([a[0,0], a[1,1], a[2,0]]))

# when using integer array indexing we can reuse the same element from source array
print(a[[0,0], [1,1]])
# the above example is similar to:
print(np.array([a[0, 1], a[0, 1]]))

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
print(a,'\n')

# create an array of indices
b = np.array([0,2,0,1])

# select one element from each row using the indices in b
print(a[np.arange(4), b], '\n')

# mutate one element from each row using the indices in b
a[np.arange(4), b] += 10
print(a)

# find elements of a that are bigger than 2;
# this returns a numpy array of booleans of the same shape as a, where each slot of bool_idx tells whether that 
# element of a is greater than 2
bool_idx = (a > 2)
print(bool_idx)

# we use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the true values of bool_idx
print(a[bool_idx])

# the above can be done in a single concise statement as:
print(a[a > 2])

x = np.array([1,2])
print(x.dtype)

x = np.array([1.0, 2.0])
print(x.dtype)

x = np.array([1,2], dtype=np.int32)
print(x.dtype)

x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[5,6], [7,8]], dtype=np.float64)

# elementwise sum
print(x + y, '\n')
print(np.add(x, y), '\n')

# elementwise difference
print(x - y, '\n')
print(np.subtract(x, y), '\n')

# elementwise product
print(x * y, '\n')
print(np.multiply(x, y), '\n')

# elementwise division
print(x / y, '\n')
print(np.divide(x, y), '\n')

# elementwise squareroot
print(np.sqrt(x, y),'\n')

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

v = np.array([9,10])
w = np.array([11,12])

# inner product of vectors
print(v.dot(w), '\n')
print(np.dot(v, w), '\n')

# Matrix/vector product
print(x.dot(v) ,'\n')
print(np.dot(x, v), '\n')

# Matrix/matrix product
print(x.dot(y), '\n')
print(np.dot(x, y))

# compute sum of all elements
print(np.sum(x))

# compute sum of each column
print(np.sum(x, axis=0))

# compute sum of each row
print(np.sum(x, axis=1))

print(x, '\n')
print(x.T, '\n')

# taking transpose of a rank 1 matrix does nothing
v = np.array([1,2,3])
print(v, '\n')
print(v.T)

# we will add vector y to each row of matrix x ; storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1,0,1])

# create a matrix y with the same shape as x
y = np.empty_like(x)

# add vector v to each row of matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v
    
print(y)

# stack 4 copies of v on top of each other
vv = np.tile(v, (4,1))
print(vv, '\n')

y = x + vv

print(y)

# add v to each row of x using broadcasting
y = x + v
print(y)

# compute outer product of vectors
v = np.array([1,2,3])
w = np.array([4,5])
# to compute outer product we first reshape v to be a column vector of shape (3,1); we then broadcast it against w
# to yield an output of shape (3,2), which is the outer product of v and w.
print(np.reshape(v, (3,1))*w, '\n')

# add a vector to each row of a matrix
x = np.array([[1,2,3],[4,5,6]])
# x has shape (2,3) and v has shape (3,) so they brodcast to (2,3)
print(x + v, '\n')

# add a vector to each column of a matrix
# x has shape (2,3) and w has shape (2,).
# if we transpose x then it has shape (3,2) and can be broadcast against w to yield a result of shape (3,2).
# transposing this matrix yields a result of (2,3) which is the matrix x with the vector w added to each column.
print((x.T + w).T, '\n')

# another solution is to reshape w to be a column vector of shape (2,1). We can broadcast it directly against x to 
# yield the same result
print(x + np.reshape(w,(2,1)), '\n')

# Multiply a matrix by a constant:
# x has shape (2,3), numpy treats scalars as arrays of shape ();
# these can broadcast together to shape (2,3)
print(x*2)


#Lets define two python lists and few constants
a = [1,2,3,4,5]
b = [9,8,7,6,5]
random = ['a','b','c',2,5,10,True,False,"hello","world"]
const1 = 10
const2 = 5
const3 = 2
random

#Try them yourself !
print(len(a))
print(a+b)
for x in a:print(x)
print([const1]*const2)
print(const3 in a)

import numpy as np
#the 'as' creates an alias for the package being imported.

x = np.array(a)
y = np.array(b)
print(x)
y 

z = np.array([a,b])
print(z)

c = ['a','b','c','d']
d = [True,False,True,True,False]
arr = np.array([a,b,c,d])
print(arr)

print(x+10)

print(z*2)
print(np.array(y/10))
print(z-20)
print(z**2) #Exponentiation

print(arr*2)

print("Zeros")
q = np.zeros((2,2))   # Create an array of all zeros
print(q)              

print("Ones")
w = np.ones((1,2))    # Create an array of all ones
print(w)              

print("Full")
e = np.full((2,2), 7.5)  # Create a constant array
print(e)      

print("Arange")
y = np.arange(10) #create an array of 10 elements, which will start with a value of 0, and increment by 1
print(y)

print("Eye")
r = np.eye(2)         # Create a 2x2 identity matrix
print(r)              

print("Random")
t = np.random.random((2,2))  # Create an array filled with random values
print(t)

print(np.delete(c,1))

x1 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
x2 = x1[:3, 1:3]
print("Original Array")
print(x1)
print("Subset of array")
print(x2)

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]])) 

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)


# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print("The Created Arrays")
print("x : \n" + str(x))
print("y : \n" + str(y))
print()
print("Addition")
print(x + y)
print(np.add(x, y))
print()
print("Subtraction")
print(x - y)
print(np.subtract(x, y))
print()
print("Multiplication")
print(x * y)
print(np.multiply(x, y))
print()
print("Division")
print(x / y)
print(np.divide(x, y))
print()
print("Square Root")
print(np.sqrt(x))

print(np.concatenate((x,y))) # Simply combines the arrays.

print("Default parameters")
print(np.append(x,y)) #this appends array y to x
print("Changing the Axis for the merge: axis = 0")
print(np.append(x,y,axis=0))
print("Changing the Axis for the merge: axis = 1")
print(np.append(x,y,axis=1))

print(x.T)
print(np.transpose(x))

z = np.array([1,2,3,4,2,5,3,7,8,8,6,4])
print(np.unique(z))


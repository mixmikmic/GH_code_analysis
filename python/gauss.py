from __future__ import print_function, division

import thinkstats2
import thinkplot

import pandas as pd
import numpy as np

from fractions import Fraction

get_ipython().magic('matplotlib inline')

def scalar_product(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum(x * y)

scalar_product([1,2,3], (4,5,6))

scalar_product([1,2,3], 2)

scalar_product([1,2,3], [2])

try:
    scalar_product([1,2,3], (4,5,6,7))
except ValueError as e:
    print(e)

class ArrayWrapper:
    def __init__(self, array):
        self.array = np.asarray(array)
        
    def __eq__(self, other):
        return np.array_equal(self.array, other.array)
    
    def __add__(self, other):
        return self.__class__(self.array + other.array)
    
    def __sub__(self, other):
        return self.__class__(self.array - other.array)
    
    def __str__(self):
        return str(self.array)
    
    def __repr__(self):
        return '%s(\n%s)' % (self.__class__.__name__, str(self.array))
    
    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, index):
        return self.array[index]
    
    def __setitem__(self, index, elt):
        self.array[index] = elt
        
    @property
    def t(self):
        return self.__class__(self.array.transpose())
    
class Vector(ArrayWrapper):
    def __mul__(self, other):
        return scalar_product(self.array, other.array)

def random_array(*shape):
    return np.random.randint(1, 10, shape)

x = Vector(random_array(3))
x

x[0], x[1], x[2]

x[1] += 1

for elt in x:
    print(elt)

y = Vector(x.array)
y

x == y

x.t

x == x.t

y = Vector(random_array(3))
y

x == y

x+y

x-y

x*y

def mm_product(array1, array2):
    dtype = np.result_type(array1, array2)
    array = np.zeros((len(array1), len(array2)), dtype=dtype)
    for i, row1 in enumerate(array1):
        for j, row2 in enumerate(array2):
            array[i][j] = scalar_product(row1, row2)
    return array

class Matrix(ArrayWrapper):
    
    def __mul__(self, other):
        return self.__class__(mm_product(self.array, other.t.array))
    
    def __truediv__(self, other):
        return self.__class__(np.linalg.solve(self.array, other.array.flat))

A = Matrix(random_array(3, 3))
A

len(A)

for row in A:
    print(row)

B = Matrix(random_array(3, 3))
B

A+B

A-B

A*B

A.array.dot(B.array)

x = Vector(random_array(3))
x

A*x

def mv_product(A, x):
    dtype = np.result_type(A, x)
    array = np.zeros(len(A), dtype=dtype)
    for i, row in enumerate(A):
        array[i] = scalar_product(row, x)
    return Vector(array)

mv_product(A.array, x.array)

A.array.dot(x.array)

x = Matrix(random_array(3, 1))
x

x == x.t

x.t * x

x * x.t

x * x

A * x

A.array.dot(x.array)

scalar = Matrix([[2]])
scalar

scalar == scalar.t

scalar * scalar

x * scalar

A * scalar

b = A * x
b

b.array

np.linalg.solve(A.array, b.array)

print(A / b)

A.array.shape

b.array.shape

m = np.hstack([A.array, b.array]).astype(Fraction)
print(m)

m[1] -= m[0]
print(m)

m[:, :-1]

m[:, -1]

def solve_augmented(m):
    m = m.astype(float)
    return np.linalg.solve(m[:, :-1], m[:,-1])

print(solve_augmented(m))

row1 = 0
row2 = 1
col = 0
pivot = m[row1, col]
victim = m[row2, col]
m[row1], pivot, victim, m[row1] * Fraction(victim, pivot)

m[row2] -= m[row1] * Fraction(victim, pivot)
print(m)

def clobber(m, row1, row2, col):
    pivot = m[row1, col]
    victim = m[row2, col]
    m[row2] -= m[row1] * Fraction(victim, pivot)

clobber(m, 0, 2, 0)
print(m)

clobber(m, 1, 2, 1)
print(m)

m[2] /= m[2,2]
print(m)




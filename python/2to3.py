import sys
print(sys.version)

# python2 has list comprehensions
[x ** 2 for x in range(5)]

# python3 has dict comprehensions!
{str(x): x ** 2 for x in range(5)}

# and set comprehensions
{x ** 2 for x in range(5)}

# magic dictionary concatenation
some_kwargs = {'do': 'this', 
               'not': 'that'}
other_kwargs = {'use': 'something', 
                'when': 'sometime'}
{**some_kwargs, **other_kwargs}

# unpacking magic
a, *stuff, b = range(5)
print(a)
print(stuff)
print(b)

# native support for unicode
s = 'Το Ζεν του Πύθωνα'
print(s)

# unicode variable names!
import numpy as np
π = np.pi
np.cos(2 * π)

# infix matrix multiplication
A = np.random.choice(list(range(-9, 10)), size=(3, 3))
B = np.random.choice(list(range(-9, 10)), size=(3, 3))
print("A = \n", A)
print("B = \n", B)

print("A B = \n", A @ B)
print("A B = \n", np.dot(A, B))

s = 'asdf'
b = s.encode('utf-8')
b

b.decode('utf-8')

# this will be problematic if other encodings are used...
s = 'asdf'
b = s.encode('utf-32')
b

b.decode('utf-8')

# shouldn't change anything in python3
from __future__ import print_function, division

print('non-truncated division in a print function: 2/3 =', 2/3)


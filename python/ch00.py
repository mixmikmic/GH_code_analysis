import datetime

# print today's date
print(datetime.date.today())

# input variables
a = 4.5
b = 5.1

# simple equation
c = a + b

# output result
print(c)

import math

# input variables
a = 6
b = 8

# somewhat more complicated formula
c = math.sqrt(a**2 + b**2)

# output result
print(c)

# input variables
n = 3
x = 4
x0 = 1
L = 6

c = math.sin(n * math.pi * (x-x0) / L)

print(c)

import numpy as np

# define row and column-based vectors
r = np.array([2, 4, 6])
c = np.array([1, 3, 5])
c.shape = (3,1)

# print the vectors
print("Row-based vector:")
print(r)
print("")
print("Column-based vector:")
print(c)

# print the second element of each vector
print("Second element of row-based vector:")
print(r[1])
print("")
print("Second element of column-based vector:")
print(c[1][0])

# build matrix
M = np.array([(1,2,3),(4,5,6),(7,8,9)])

# output matrix and its transpose
print("Matrix M:")
print(M)
print("")
print("Matrix M transposed:")
print(np.transpose(M))

# build matrices
M = np.array([(1,0,2), (0,1,0), (2,0,1)])
N = np.array([(1,0,-1), (0,2,0), (1,0,3)])

# sum the matrices M and N
S = M + N

print("M + N =")
print(S)

# build matrices
M = np.array([(1,1,2,2),(2,2,3,3)])
N = np.array([(1,2),(2,1),(1,2),(2,1)])

# multiply the matrices M and N
Q = M.dot(N)

print("M * N =")
print(Q)

# build 3x3 identity matrix
I = np.eye(3)

print("3x3 Identity matrix:")
print(I)

# build vectors and matrices
a = np.array([1,3,5])
b = np.array([2,4,6])
M = np.array([(1,0,2),(0,1,0),(2,0,1)])
N = np.array([(1,0,-1),(0,2,0),(-1,0,3)])

s = np.vdot(a,b)
print(s)

T = a.reshape(3,1) * b
print(T)

c = M.dot(a.reshape(3,1))
print(c)

P = M.dot(N)
print(P)

D = np.multiply(M, N)
print(D)

a = np.array([1, 2, 3]).reshape(3,1)
M = np.array([(1,2,3), (4,5,6), (7,8,9)])

# we need to use a second index because we're using a column-based array now
s = a[1,0]
print("s =", s)

t = M[1,2]
print("t =", t)

b = M[:,1]
print("b =", b)

c = M[2,:]
print("c =", c)

T = M[1:,1:]
print("T =", T)

from numpy.linalg import inv, det

A = np.array([(1,5,13),(2,7,17),(3,11,19)])
b = np.array([1,2,3])

B = inv(A)
print("B =")
print(B)

print("")

d = det(A)
print("d =")
print(d)

# calculate the vector c, and double-check its value
c = np.round(np.linalg.solve(A, b), decimals=10)
assert np.allclose(b, A.dot(c)), "Error: Ac != b"
print("c =")
print(c)

print("")

# calculate the matrix D, and double-check its value
D = np.round(np.linalg.solve(A, B), decimals=10)
assert np.allclose(B, A.dot(D)), "Error: AD != B"
print("D =")
print(D)

M = np.array([(1,2,0), (2,2,0), (0,0,4)])
w, v = np.linalg.eig(M)

print("Eigenvalues: ", w)

print("")

print("Eigenvectors: ")
print(v)

assert v[:,0].dot(v[:,0]) == 1, "Orthonomality failed: First eigenvector not length 1"
assert v[:,1].dot(v[:,1]) == 1, "Orthonomality failed: Second eigenvector not length 1"
assert v[:,2].dot(v[:,2]) == 1, "Orthonomality failed: Third eigenvector not length 1"
assert v[:,0].dot(v[:,1]) == 0, "Orthonomality failed: First and second eigenvectors not orthogonal"
assert v[:,0].dot(v[:,2]) == 0, "Orthonomality failed: First and third eigenvectors not orthogonal"
assert v[:,1].dot(v[:,2]) == 0, "Orthonomality failed: Second and third eigenvectors not orthogonal"

# create matrix and initialize vector
M = np.array([(1,4,7), (2,5,8), (3,6,9)])
a = np.zeros(3)

# loop through items along M's diagonal - copy values to vector 'a'
for i in range(0, 3):
    a[i] = M[i,i]

# print vector a
print("a = ", a)

# re-initialize the vector
a = np.zeros(3)

# extract diagonal
a = np.diagonal(M)

print("a = ", a)

import pandas as pd

df = pd.read_csv("../data/global_temp.txt", delim_whitespace=True, header=None, names=["Year", "TempChange"])
df.head()

import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

df.plot(x='Year', y='TempChange', figsize=(10,6))
plt.xlim(1965, 2010)
plt.ylim(-0.5, 1.0)
plt.xlabel('Calendar Year')
plt.ylabel('Temperature Anomaly ($^{\circ}C$)')
plt.title('Global Temperature Data 1965-2010')

# get number of rows in data frame
n = df.shape[0]

print ("The global temperature data begins for the year %s and ends for the year %s." % (df.Year[0], df.Year[n-1]))


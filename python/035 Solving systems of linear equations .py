import numpy as np
from numpy.linalg import inv, det

A = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
B = np.array([6, -4, 27])

print '[A]'
print A

print
print '[B]'
print B

print 'X의 값은:', inv(A).dot(B)


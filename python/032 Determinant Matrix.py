import numpy as np
from numpy.linalg import inv, det

A = np.array([[4, 6], [3, 8]], dtype=np.float32)

print 'Matrix A'
print A
print 
print 'The determinant of a Matrix A:', det(A)
print '4*8 - 3*6 = 32 - 18 = 14'

C = np.array([[6,  1,  1], 
              [4, -2,  5], 
              [2,  8,  7]])
print 'Matrix C'
print C
print 
print 'The determinant of a Matrix C:', det(C)

np.array([])


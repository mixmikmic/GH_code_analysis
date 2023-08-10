import numpy as np

M = np.arange(2*4*3).reshape([2,4,3])
M

np.arange(100)[80:85]

M[:, 1:, 0 : 2]

M[1,1,1]

M + M / 2.0

M[:,0,1].reshape((2,1,1))

M[:,0,1].reshape((2,1,1)) + M

M[1,:,:]

s = M[1,:,0:2]
s

s * 10

M

s[:] = 0
s

M

S = M[0,:,:].copy()

M

S

S[:] = 10

S

M

M == 5

M % 2 == 0

evens = M[M % 2 == 0]



evens = evens ** 2 + 10

evens

M

M[M % 2 == 0] = evens

M




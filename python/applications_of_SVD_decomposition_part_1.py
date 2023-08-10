from __future__ import division

import numpy as np
from numpy import linalg as LA

A = np.array([[1, 2, 0, 0, 2, 3, -1, -2]]).reshape((4, 2))
print A

n, m = A.shape
rank_A = LA.matrix_rank(A)

print "rows:", n, "columns:", m
print "the rank of A is", rank_A

U, d, V_T = LA.svd(A)
number_of_singular_values = len(d)
print "number of singular values:", number_of_singular_values

D = np.concatenate((np.diag(d), np.zeros((n - number_of_singular_values, m))))
A_restored = np.dot(U, np.dot(D, V_T))

np.allclose(A, A_restored, rtol=1e-14, atol=1e-15)

np.abs(A - A_restored)

B = np.array([[1, 2, 0, 0, 2, 4, -1, -2]]).reshape((4, 2))
print B

rank_B = LA.matrix_rank(B)
distance = LA.norm(A - B, "fro")

print "the rank of B is", rank_B
print "the Frobenius-norm of the difference matrix A-B is", distance

U_1 =  U[:, 0:rank_B]
D_1 = D[0:rank_B, 0:rank_B]
V_T_1 = V_T[0:rank_B, :]

A_1 = np.dot(U_1, np.dot(D_1, V_T_1))
print "the best (that is the closest) rank 1 approximation of A is \n", A_1

dist_from_A_1 = LA.norm(A - A_1, "fro")
print "The Frobenius-norm of the difference matrix A-A_1 is", dist_from_A_1,       "which is smaller that in our previous naive attempt."


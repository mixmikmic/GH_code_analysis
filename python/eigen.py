import numpy as np    # python module useful in handling arrays, vectors, matrices etc.
from numpy import linalg as LA   # for eigenvalue and eigenvector computation

A = np.matrix('4 -1 5; 0 6 0; 1 -2 0')    # declaring a numpy matrix
print(A)

EigenValues, EigenVectors = LA.eig(A)

EigenValues

EigenVectors

print ( "eingenvector for eigenvector", EigenValues[0], "is" , EigenVectors[0] )
print ( "eingenvector for eigenvector", EigenValues[1], "is" , EigenVectors[1] )
print ( "eingenvector for eigenvector", EigenValues[2], "is" , EigenVectors[2] )




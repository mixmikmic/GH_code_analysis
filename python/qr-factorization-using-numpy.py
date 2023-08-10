import numpy as np
import math

def project_parallel(b, v, epsilon= 1e-15):
    '''
    Projection of b along v
    
    Inputs:
        - b: 1-d array
        - v: 1-d array
        - epsilon: threshold for filling 0 for squared norms
    
    Output:
        - 1-d array: the projection of b parallel to v
    '''
    sigma = (np.dot(b,v)/np.dot(v,v)) if np.dot(v,v) > epsilon else 0
    return (sigma*v)

def project_orthogonal(b, A):
    '''
    Project b orthogonal to row vectors of A

    Inputs:
        - b: 1-d array
        - A: 2-d array

    Output: the projection of b orthogonal to the row vectors of A
    '''
    for v in A:
        b = b - project_parallel(b, v)
    return b

A = np.array([[1, 0, 0], [0, 1, 0]])
print (A, '\n')
b = np.array([1, 1, 1])
print (b)
project_orthogonal(b, A)

def orthogonalize(A):
    '''
    Orthogonalizes the row vectors of A.
    Row i of the output matrix is the projection of row i of the input
    matrix orthogonal to the space spanned by all previous rows in the 
    input matrix.
    
    Input: 2-d array
    
    Output: 2-d array of mutually orthogonal row vectors spanning the
            same space as the original row space'''
    
    orth_list = np.zeros(A.size)
    orth_list.shape = A.shape
    for i, v in enumerate(A):
        orth_list[i] = (project_orthogonal(v, orth_list))
    return orth_list

A = np.array([[8, -2, 2], [4, 2, 4]])
orthogonalize(A)

def orthonormalize(A):
    '''
    Orthonormalizes the row vectors of A
    
    Input: 2-d array
    
    Output: 2-d array of orthonormalized vectors
    '''
    return np.stack([v/math.sqrt(sum(v**2)) for v in orthogonalize(A)])

orthonormalize(A)

def QR_factorize(A):
    '''
    Factorizes A into orthonormal matrix Q and triangular matrix R
    
    Input: a 2-d array with linearly independent columns
    
    Outputs:
        Q = a matrix of orthonormal colum vectors that span the same column
            space as the input
        R = a triangualr matrix of vectors such that Q*R = A
    '''
    Q = orthonormalize(A.T)
    R = np.dot(Q, A)
    return Q.T, R

A = np.array([[4, 8, 10], [3, 9, 1], [1, -5, -1], [2, -5, 5]])
Q, R = QR_factorize(A)

print('Original matrix A:')
print(A, '\n')
print('Q:')
print(Q, '\n')
print('R:')
print(R, '\n')
print('Q*R')
print(np.dot(Q, R), '\n')

Q, R = np.linalg.qr(A)
print('Q:')
print(Q, '\n')
print('R:')
print(R, '\n')
print('Q*R')
print(np.dot(Q, R), '\n')

A = np.random.rand(1000, 1000)

get_ipython().run_cell_magic('time', '', 'Q, R = QR_factorize(A)')

get_ipython().run_cell_magic('time', '', 'Q, R = np.linalg.qr(A)')


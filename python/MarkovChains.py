import os, sys
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf



# transition matrix
P = np.array([[2/3, 1/3], 
              [1/3, 2/3]])

# initial distribution can be anything that sums up to 1
pi0 = np.array([0.5, 0.5])

# compute stationary distribution - power method
np.dot(pi0, np.linalg.matrix_power(P, 50))

# transition matrix
P = np.array([[3/4, 1/4], 
                [1/3, 2/3]])

# initial distribution can be anything that sums up to 1
pi0 = np.array([0.5, 0.5])

# compute stationary state - power method
np.dot(pi0, np.linalg.matrix_power(P, 50))

# some random 5x5 transition matrix
P = np.random.rand(5, 5)
P /= P.sum(axis = 1)[:, np.newaxis] # normalization along axis 1

# compute stationary score - power method
pi0 = np.random.rand(5)
pi0 /= pi0.sum()
a = np.dot(pi0, np.linalg.matrix_power(P, 50))
print(a)

# compute stationary state - eigen decomposition
L, Q = np.linalg.eig(P.T)

# pick eigenvector whose corresponding eigenvalue is closest to 1
b = Q[:, np.argmin(abs(L - 1.0))].real

# normalize into a probability distribution
b /= b.sum()
print(b)

np.allclose(a, b)

# compute stationary state - power method

def mat_power(M, n):
    """ Construct a graph that raises square matrix M to n-th power where n>=1
    This generates a computational graph with space complexity O(log(n)).
    """
    
    assert n >= 1
    
    # trivial cases
    
    if n == 2:
        return tf.matmul(M, M)
    elif n == 1:
        return M
    
    # divide and conquer
    A = mat_power(M, n//2)
    A2 = tf.matmul(A, A)
    
    if n &1: # odd power
        return tf.matmul(A2, M)
    else: # even power
        return A2

def get_stationary_state(P):
    pi0 = tf.constant(np.ones((1, len(P)))/len(P))
    transition_matrix = tf.constant(P)
    stationary_state = tf.squeeze(tf.matmul(pi0, mat_power(transition_matrix, 50)))
    with tf.Session() as sess:
        return sess.run(stationary_state)

a = get_stationary_state(P)
print(a)

np.allclose(a, b)


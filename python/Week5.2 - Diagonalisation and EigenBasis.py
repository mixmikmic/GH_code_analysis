import numpy as np
import numpy.linalg as nl

A = np.array([[2,0], [0,2]])

A @ A @ A

def getDiagonal(T, C):
    T = np.array(T)
    C = np.array(C)
    Cinv = nl.inv(C)
    return Cinv @ T @ C

T = np.array([[6,-1], [2,3]])

C = np.array([[1,1], [1,2]])

D = getDiagonal(T, C)
D

getDiagonal([[2,7],[0,-1]], [[7,1], [-3,0]])

getDiagonal([[1,0], [2,-1]], [[1,0], [1,1]])

C = np.array([[1, 2], [0,1]])

nl.inv(C)

C @ np.array([[10, 0], [0, 10]]) @ nl.inv(C)

T = np.array([[6, -1], [2,3]])

T @ T @ T

T = np.array([[2,7], [0 , -1]])

T @ T @ T

T = np.array([[1,0], [2,-1]])

T ** 5 # Does not work because T is a np.array

T**5

T @ T @ T @ T @ T

nl.matrix_power(T, 5)

T = np.matrix([[1,0], [2,-1]])

T ** 5 # T has to be a matrix for that to work




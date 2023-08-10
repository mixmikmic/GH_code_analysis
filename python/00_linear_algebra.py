def vector_add(v, w):
    """adds corresponding elements"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]

a = [5, 5]
b = [10, 10]

vector_add(a, b)

def vector_subtract(v, w):
    """subtracts corresponding elements"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]

vector_subtract(a, b)

from functools import reduce

def vector_sum(vectors):
    '''sums all corresponding elements'''
    return reduce(vector_add, vectors)


#def vector_sum(vectors):
#    """sums all corresponding elements"""
#    result = vectors[0]
#    for vector in vectors[1:]:
#        result = vector_add(result, vector)
#    return result

c = [15, 15]
d = [20, 20]

vectors = [a, b, c, d]

vector_sum([a, b, c, d])

def scalar_multiply(c, v):
    """multiplies all elements of a vector by a scalar:
    c is a number, v is a vector"""
    return [c * v_i for v_i in v]

scalar_multiply(5, a)

def vector_mean(vectors):
    """compute the vector whose ith element is the mean
    of the ith element of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
    

vectors

vector_mean(vectors)

def dot(v, w):
    """v_1 * w_1 + v_2 * w_2 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

dot(a, b)

def sum_of_squares(v):
    """v_1 * v_1 + v_2 * v_2 + ... + v_n * v_n"""
    return dot(v, v)

sum_of_squares(a)

import math

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

magnitude(a)

def squared_distance(v, w):
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(vector_subtract(v, w))

a, b

e = 10, 10
f = 20, 20

squared_distance(e, f)

def distance(v, w):
    return magnitude(vector_subtract(v, w))


#def distance(v, w):
#    return math.sqrt(squared_distance(v, w))

distance(e, f)

A = [[1, 2, 3],
     [4, 5, 6]] 

B = [[1, 2],
     [3, 4],
     [5, 6]] 

def shape(A):
    """Return the number of rows and number of cols in a matrix"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

shape(A)

shape(B)

def get_row(A, i):
    """Return a specified row"""
    return A[i]

get_row(A, 1)

def get_col(A, j):
    """Return a specified column"""
    return [A_i[j] for A_i in A]

get_col(B, 1)

def make_matrix(num_rows, num_cols, entry_fn):
    """Returns a matrix with num_rows x num_cols
    whose (i,j)th entry is generated with the
    entry_fn(i, j)"""
    
    return [[entry_fn(i, j) for j in range(num_cols)]
                                for i in range(num_rows)]

def outputs(i, j):
    return i * j

make_matrix(3, 3, outputs)

def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal)
identity_matrix


friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

friendships[0][2]

friendships[0][8]

friends_of_five = [i for i, is_friend in enumerate(friendships[5]) if is_friend]

friends_of_five



































from functools import reduce

reduce(lambda x, y: x+y, [1, 2, 3, 4])




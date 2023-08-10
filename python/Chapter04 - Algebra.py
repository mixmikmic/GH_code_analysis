from string import Template
from __future__ import division
import math

v = [1, 2]
w = [3, 4]
x = [5, 6]

addition = lambda (x, y): x + y
subtraction = lambda (x, y): x - y
product = lambda (x, y): x * y

def vector_operation(v, w, op):
    return map(op, zip(v, w))

# We can make usage of this abstract function that just operates on tuples
def add(v, w): 
    return vector_operation(v, w, addition)
def sub(v, w): 
    return vector_operation(v, w, subtraction)

# sum a list of vectors
def vector_sum(vectors):
    return reduce(add, vectors)

# multiply a scalar by a vector
def scalar_multiply(c, v):
    return [c * vi for vi in v]

# mean of a list of vectors
def vector_mean(vectors):
    return scalar_multiply(1 / len(vectors), vector_sum(vectors))

# dot product: should be sum(vi * wi) given v and w from Rn (length of the vector projected by v onto w)
def dot(v, w):
    return sum(vector_operation(v, w, product))

def sum_of_squares(v):
    return dot(v, v)

# magnitude: what's the vector length in space?
def norm(v):
    return math.sqrt(sum_of_squares(v))

# distance between two vectors 
def distance(v, w):
    # could've been sqrt(sum_squares(sub(v, w)))
    return norm(sub(v, w))

temp = Template(
    """
    Given: 
        v = [1, 2]
        w = [3, 4]
        x = [5, 6]
        
    - Addition v + w: 
        $res_add
            
    - Subtract v - w: 
        $res_sub
            
    - Summing [v, w, x]: 
        $res_sum
            
    - Multiply v by 3: 
        $res_mul
        
    - Mean of [v, w, x]:
        $res_mean
        
    - Dot product v * w:
        $res_dot
    
    - Norm sqrt(sum(vi^2)):
        $res_norm
        
    - Distance between v and w:
        $res_dist
    """)

print(temp.substitute(
    res_add  = add(v, w),
    res_sub  = sub(v, w),
    res_sum  = vector_sum([v, w, x]),
    res_mul  = scalar_multiply(3, v),
    res_mean = vector_mean([v, w, x]),
    res_dot  = dot(v, w),
    res_norm = norm(v),
    res_dist = distance(v, w)
))

A = [[1, 2, 3],
     [4, 5, 6]]

B = [[1, 2],
     [3, 4],
     [5, 6]]

# how many rows and columns does this matrix have?
def shape(A):
    num_rows = len(A)
    num_columns = len(A[0]) if A else 0 # may be empty
    return num_rows, num_columns

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in A]

def create_matrix(num_rows, num_cols, f):
    """Returns a (m, n) matrix with f(i, j) applied on each element"""
    return [[f(i, j) 
             for i in range(num_rows)]
            for j in range(num_cols)]

# Let's create a 5x5 identity

def is_diagonal(x, y):
    if x == y: 
        return 1
    else: 
        return 0

print create_matrix(5, 5, is_diagonal)


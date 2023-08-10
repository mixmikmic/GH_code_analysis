from snf import *

import numpy as np

from datetime import datetime

# times = []
# for i in range(20):
#     matrix = np.random.random_integers(0, 1, (757,966))
#     start = datetime.now()
#     put_in_snf(matrix)
#     end = datetime.now()
#     duration = (end - start).total_seconds()
#     print(duration)
#     times.append(duration)

times = []
for i in range(20):
    matrix = np.random.random_integers(0, 1, (1000, 1000))
    start = datetime.now()
    reduce_matrix(matrix)
    end = datetime.now()
    duration = (end - start).total_seconds()
    print(duration)
    times.append(duration)

for i in range(20):
    matrix = np.random.random_integers(0, 1, (100,200))
    matrix2 = matrix.copy()
    put_in_snf(matrix2)
    matrix, _, _ = reduce_matrix(matrix)
    print(np.array_equal(matrix, matrix2 % 2))

truth = []
for i in range(100):
    matrix = np.random.random_integers(0, 1, (100,250))
    matrix_2 = matrix.copy()
    x,y,z = reduce_matrix(matrix)
    a,b,c = reduce_matrix_iter(matrix_2)
    print(np.array_equal(a,x))
    truth.append(np.array_equal(a,x))


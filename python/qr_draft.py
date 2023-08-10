import numpy as np
from functools import reduce

M = 14
N = 4
init_matrix = np.random.rand(14, 3)
STEP = 5

def get_matrix_by_chunck(matrix, step=STEP):
    cur = 0
    while cur < len(matrix):
        yield matrix[cur:cur+step]
        cur = cur + step

chunks = list(get_matrix_by_chunck(init_matrix))

def cal_r(m1, m2):
    return np.linalg.qr(np.concatenate((m1, m2)))[1]

distributed_r = reduce(cal_r, chunks)
distributed_q = np.dot(init_matrix, np.linalg.inv(distributed_r))

np.dot(distributed_q, distributed_r)

init_matrix




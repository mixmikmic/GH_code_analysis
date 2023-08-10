import numpy as np
from numpy.linalg import norm, cond, solve
import time
import matplotlib.pyplot as plt

tic = lambda: time.time()
toc = lambda t: time.time() - t

print('{:^5} {:^5}   {:^11} {:^11}\n{}'.format('m', 'n', 'solve(A,b)', 'dot(inv(A), b)', '-'*40))

for m in [1, 100]:
    for n in [50, 500]:
        A = np.random.rand(n, n)
        b = np.random.rand(n, 1)

        tt = tic()
        for j in range(m):
            x = solve(A, b)

        f1 = 100 * toc(tt)

        tt = tic()
        Ainv = np.linalg.inv(A)
        for j in range(m):
            x = np.dot(Ainv, b)

        f2 = 100 * toc(tt)
        print(' {:3}   {:3} {:11.2f} {:11.2f}'.format(m, n, f1, f2)) 


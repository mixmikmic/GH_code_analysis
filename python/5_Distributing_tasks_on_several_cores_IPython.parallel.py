import numpy as np
# ipyparallel was Ipython.parallel before IPython 4.0
from ipyparallel import Client

rc = Client()

rc.ids

get_ipython().magic('px import os, time')

get_ipython().magic('px print(os.getpid())')

get_ipython().run_cell_magic('px', '--targets :-1', '    print(os.getpid())')

v = rc.load_balanced_view()

def sample(n):
    import numpy as np
    # Random coordinates.
    x,y = np.random.rand(2,n)
    # Square distances tot the origin.
    r_square = x ** 2 + y ** 2
    # Number of points in the quarter disc.
    return (r_square <= 1).sum()

def pi(n_in, n):
    return 4. * float(n_in) / n

n = 100000000

pi(sample(n),n)

get_ipython().magic('timeit pi(sample(n),n)')

args = [n // 100] * 100

ar = v.map(sample, args)

ar.ready(), ar.progress

ar.elapsed, ar.serial_time

get_ipython().magic('debug')

pi(np.sum(ar.result),n)


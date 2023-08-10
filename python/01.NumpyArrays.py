import numpy as np

x = np.arange(5)
x

y = np.ones(5)
y

x - y * 3

np.array([3,3,"string",5,5])

_17[3]

sorted(x - y * 3)

x.dtype

def py_add(a, b):
    c = []
    for i in xrange(0,len(a)):
        c.append(1.324 * a[i] - 12.99*b[i] + 1)
    return c

def np_add(a, b):
    return 1.324 * a - 12.99 * b + 1

a = np.arange(1e6)
b = np.random.randn(1e6)
len(a)

b[0:20]

get_ipython().magic('timeit py_add(a,b)')

get_ipython().magic('timeit np_add(a,b)')




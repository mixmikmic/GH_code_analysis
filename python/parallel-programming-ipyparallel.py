get_ipython().system('pip install ipyparallel')

get_ipython().system('ipcluster start -n 4')

from ipyparallel import Client
cluster = Client()
c = cluster[:]

squares = c.map_sync(lambda x: x**2, range(1,28))
print('squares:',squares)

import numpy as np
y = np.random.randn(50,50)
x = np.matrix(y)
def matmul(x, y):
    return x*y
res = c.apply_sync(matmul, x, x)

@c.parallel(block=True)
def arrmul(x,y):
    return x*y
res = arrmul(y, y)

get_ipython().system('ipcluster stop')


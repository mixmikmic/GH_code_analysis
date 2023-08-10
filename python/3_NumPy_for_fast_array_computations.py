import random
import numpy as np

get_ipython().magic('precision 3')

n = 1000000
x = [random.random() for _ in range(n)]
y = [random.random() for _ in range(n)]

x[:3], y[:3]

z = [x[i] + y[i] for i in range(n)]
z[:3]

get_ipython().magic('timeit [x[i] + y[i] for i in range(n)]')

xa = np.array(x)
ya = np.array(y)

za = xa + ya
za[:3]

get_ipython().magic('timeit xa + ya')


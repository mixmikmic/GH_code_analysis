get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import numpy as np

l = [1,2,3,4]

d1= np.array(l)
d1

d2= np.array([(1,2,3),[4,5,6],[7,8,'A']])
d2

type(d2)

d1.ndim

d2.ndim

d1.shape

d2.shape

d1.dtype

np.arange(9)

np.zeros(3)

np.zeros((4,4))

np.ones((4,5))

np.empty(5)

print (np.linspace(1,10,5,endpoint = False))
print (np.arange(1,10,1))

np.eye(3)

np.random.rand(3,4)



np.random.randn(4,5)

np.random.randint(10,50,200)


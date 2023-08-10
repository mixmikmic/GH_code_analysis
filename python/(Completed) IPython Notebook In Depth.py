1 + 1

_ * 100

_ + __

_ + __ ** ___

_3

Out[3]

Out

In[2]

In

get_ipython().magic('history')

my_variable = 4

get_ipython().magic('pinfo numpy.random')

def myfunc(x):
    return x ** 2

get_ipython().magic('pinfo myfunc')

get_ipython().magic('pinfo2 myfunc')

get_ipython().magic('psearch numpy.*exp*')

get_ipython().magic('pinfo %timeit')

import numpy as np
x = np.random.rand(1000000)

get_ipython().magic('timeit x.sum()')

L = list(x)

get_ipython().magic('timeit sum(L)')

get_ipython().run_cell_magic('timeit', '', '\ny = x + 1\nz = y ** 2\nq = z.sum()')

get_ipython().run_cell_magic('file', 'myscript.py', '\ndef foo(x):\n    return x ** 2\n\nz = foo(12)\n\nprint(foo(14))')

get_ipython().magic('run myscript.py')

z

foo(2)

get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.random.rand(100));

get_ipython().magic('lsmagic')

get_ipython().magic('magic')

get_ipython().magic('pinfo %debug')

get_ipython().system('ls')

get_ipython().system('pwd')

contents = get_ipython().getoutput('ls')

contents

get_ipython().system('cat {contents[4]}')

for filename in contents:
    if filename.endswith('.py'):
        print(filename)
        get_ipython().system('head -10 {filename}')


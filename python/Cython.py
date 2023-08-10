x = 5 #int 
x = 5.425 #float 
x = [1, 2, 3, 4, 5] #list 
x = 'hello' #string

get_ipython().magic('load_ext Cython')

get_ipython().run_cell_magic('cython', '', 'cimport numpy as np')

def test(x):
    '''useless function to increment y x times'''
    y = 0
    for i in range(x):
        y += i
    return y

get_ipython().run_cell_magic('cython', '-a ', "cpdef int cy_test(int x):\n    '''useless function to increment y x times'''\n    cdef int y = 0\n    for i in range(x):\n        y += i\n    return y")

get_ipython().run_cell_magic('cython', '-a ', "cpdef int cy_test(int x):\n    '''useless function to increment y x times'''\n    cdef int y = 0\n    cdef int i = 0\n    for i in range(x):\n        y += i\n    return y")

import timeit

cy = timeit.timeit('cy_test(50)', setup = "from __main__ import cy_test", number=10000)
py = timeit.timeit('test(50)', setup = "from __main__ import test", number=10000)

print ('Cython takes {}s per iteration'.format(cy))
print('Python takes {}s per iteration'.format(py))
print('Cython is {}x faster'.format(py/cy))

import cy_test

cy_test.cy_test(5)




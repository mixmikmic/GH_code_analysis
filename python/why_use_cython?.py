def fibonacci_py(n):
    a, b = 0, 1
    for _ in range(1, n):
        a, b = b, a + b
    return b

get_ipython().magic('timeit fibonacci_py(0)')

get_ipython().magic('timeit fibonacci_py(70)')

get_ipython().magic('load_ext Cython')

get_ipython().run_cell_magic('cython', '', 'def fibonacci_cy_naive(n):\n    a, b = 0, 1\n    for _ in range(1, n):\n        a, b = b, a + b\n    return b')

get_ipython().magic('timeit fibonacci_cy_naive(0)')

get_ipython().magic('timeit fibonacci_cy_naive(70)')

get_ipython().run_cell_magic('cython', '', 'def fibonacci_cy_static(n):\n    cdef int _\n    cdef int a=0, b=1\n    for _ in range(1, n):\n        a, b = b, a + b\n    return b')

get_ipython().magic('timeit fibonacci_cy_static(0)')

get_ipython().magic('timeit fibonacci_cy_static(70)')


n = 10**7

def foo(n):
    for i in range(n):
        break

foo(n)

get_ipython().magic('timeit foo(n)')

n = 10**8

get_ipython().magic('timeit foo(n)')

n = 10**9

get_ipython().magic('timeit foo(n)')

n = 10**12345

get_ipython().magic('timeit foo(n)')


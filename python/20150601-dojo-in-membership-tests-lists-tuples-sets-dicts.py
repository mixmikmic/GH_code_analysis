n = 10**7
a = list(range(n))
b = tuple(a)
c = set(a)
d = dict(zip(a, a))

5 in a, 5 in b, 5 in c, 5 in d

i = n/2

get_ipython().magic('timeit i in a')

get_ipython().magic('timeit i in b')

get_ipython().magic('timeit i in c')

730e-3 / 289e-9

get_ipython().magic('timeit i in d')

730e-3 / 295e-9


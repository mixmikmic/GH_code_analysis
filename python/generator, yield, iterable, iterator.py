(i for i in range(1000) if i % 2)

import sys

sys.getsizeof( [i for i in range(1000) if i % 2] )

sys.getsizeof( (i for i in range(1000) if i % 2) )

def generator(n):
    i = 0
    while i < n:
        yield i
        i += 1


x = [1,2,3]

x.next()

type(x)

y = iter(x)

type(y)

next(y)

next(y)

next(y)

next(y)




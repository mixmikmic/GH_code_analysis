from __future__ import print_function

def foo(n):
    for i in range(n):
        yield i*i

g = foo(4)
g

h = foo(5)

next(g)

g.next()

next(g)

for i in h:
    print(i)

next(g)

next(g)

next(g)


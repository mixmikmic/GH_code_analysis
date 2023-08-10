def foo(n):
    for i in range(n):
        yield i*i

g = foo(4)
g

h = foo(5)

g.next()

next(g)

for i in h:
    print i

g.next()

g.next()

g.next()


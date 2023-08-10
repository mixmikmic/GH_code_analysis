from itertools import count, islice, accumulate

def foo():
    yield from range(3)
    yield from ('foo', 'bar')
    yield from 'the'
    yield from accumulate(range(1, 4+1))
    yield from open('threelines.txt')
    yield from count()

list(islice(foo(), 20))


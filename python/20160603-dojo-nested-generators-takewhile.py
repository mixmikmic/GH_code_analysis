from itertools import count, takewhile

def squares():
    for i in count():
        yield i*i

list(takewhile(lambda x: x < 100, squares()))

sum(takewhile(lambda x: x < 100, squares()))


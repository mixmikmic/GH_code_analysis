import cProfile

def simple_add(a,b):
    return a+b


def some_code_to_profile():
    c = 0
    for i in xrange(10000):
        c = c + simple_add(i,i)
    return c


cProfile.run('some_code_to_profile()')


import time
def f(p, n):
    """
    Return p^{2^n} together with timing
    """
    times = []
    t = time.clock()
    for i in range(n):
        p = p*p
        t2 = time.clock()
        times.append(t2-t)
        print i, t2-t
        t = t2
    return p, times

P = ZZ['x']

x = P.gen()

p = 2*x+3

f(p, 0)

f(p,1)

p, timings = f(p, 15); timings

points(enumerate(timings))

def pone(n):
    """Returns 1+x+...+x^k"""
    return P([1]*n)

n = 23
p = pone(2^n)
get_ipython().run_line_magic('time', '_ = p*p;')

n = 24
p = pone(2^n)
get_ipython().run_line_magic('time', '_ = p*p;')

n = 25
p = pone(2^n)
get_ipython().run_line_magic('time', '_ = p*p;')


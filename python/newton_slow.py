get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import abs

a = 1.0e-10

def F(x):
    return 1/x - a

def DF(x):
    return -1/x**2

M   = 100      # maximum iterations
x   = a     # initial guess
eps = 1e-15 # relative tolerance on root

f = F(x)
for i in range(M):
    df = DF(x)
    dx = -f/df
    x  = x + dx
    e  = abs(dx)
    f = F(x)
    print "{0:6d} {1:22.14e} {2:22.14e} {3:22.14e}".format(i,x,e,abs(f))
    if e < eps * abs(x):
        break


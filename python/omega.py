get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import linspace,pi,cos
from matplotlib.pyplot import plot,legend,title

def omega(x,xp):
    f = 1.0
    for z in xp:
        f = f * (x-z)
    return f

M  = 1000
xx = linspace(-1.0,1.0,M)

N = 16
xu = linspace(-1.0,1.0,N+1)    # uniform points
xc = cos(linspace(0.0,pi,N+1)) # chebyshev points
fu = 0*xx
fc = 0*xx
for i in range(M):
    fu[i] = omega(xx[i],xu)
    fc[i] = omega(xx[i],xc)
plot(xx,fu,'b-',xx,fc,'r-')
legend(("Uniform","Chebyshev"))
title("N = "+str(N));


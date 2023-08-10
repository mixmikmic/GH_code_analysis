get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import linspace,pi,cos
from matplotlib.pyplot import plot,legend,title,grid

def omega(x,xp):
    f = 1.0
    for z in xp:
        f = f * (x-z)
    return f

def plot_omega(x):
    M  = 1000
    xx = linspace(-1.0,1.0,M)
    f  = 0*xx
    for i in range(M):
        f[i] = omega(xx[i],x)
    plot(xx,f,'b-',x,0*x,'o')
    title("N = "+str(N));
    grid(True)

N = 3
x = linspace(-1.0,1.0,N+1)
plot_omega(x)

N = 4
x = linspace(-1.0,1.0,N+1)
plot_omega(x)

N = 6
x = linspace(-1.0,1.0,N+1)
plot_omega(x)


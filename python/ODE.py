get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import zeros,exp
from matplotlib.pyplot import plot,xlabel,ylabel,legend

def ode(alpha,h,N):
    """
    h = step size
    N = number of steps to take
    """
    y = zeros(N)
    t = zeros(N)
    t[0], y[0] = 0, 1
    t[1], y[1] = h, exp(alpha*h)
    a = 2.0*alpha*h
    for i in range(1,N-1):
        y[i+1] = a*y[i] + y[i-1]
        t[i+1] = t[i] + h
    ye = exp(alpha*t)
    plot(t,y,'o-',t,ye,'r-')
    xlabel('t')
    ylabel('y')
    legend(('Numerical','Exact'),loc='upper left')

alpha = 1
h = 0.1
N = 20
ode(alpha,h,N)

alpha = -1
h = 0.1
N = 100
ode(alpha,h,N)


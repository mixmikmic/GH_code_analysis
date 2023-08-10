get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import numpy as np
from matplotlib import pyplot as plt

# construct LU decomposition
def tdma1(a,b,c):
    n = len(a)
    c[0] = c[0]/a[0]
    for i in range(1,n):
        a[i] = a[i] - b[i]*c[i-1]
        c[i] = c[i]/a[i]
    a[n-1] = a[n-1] - b[n-1]*c[n-2]
    return a,b,c

# solve
def tdma2(a,b,c,f):
    n = len(f)
    x = np.empty_like(f)
    # solve L y = f
    x[0] = f[0]/a[0]
    for i in range(1,n):
        x[i] = (f[i] - b[i]*x[i-1])/a[i]
    # solve U x = y
    for i in range(n-2,-1,-1):
        x[i] = x[i] - c[i]*x[i+1]
    return x

xmin, xmax = 0.0, 1.0
uexact = lambda x: 0.5*x*(1-x)

n = 50
h = (xmax - xmin)/n

x = np.linspace(0.0, 1.0, n+1) # Grid
f = h**2 * np.ones(n+1)        # Right hand side
f[0] = uexact(xmin); f[n] = uexact(xmax);

# Create the three diagonals
a =  2.0*np.ones(n+1)
a[0] = 1.0; a[n] = 1.0

b = -1.0*np.ones(n+1)
b[n] = 0.0

c = -1.0*np.ones(n+1)
c[0] = 0.0

# Compute LU decomposition and solve the problem
a,b,c, = tdma1(a,b,c)
u = tdma2(a,b,c,f)

ue = uexact(x) # Exact solution

plt.plot(x,ue,x,u,'o')
plt.legend(("Exact","Numerical"));


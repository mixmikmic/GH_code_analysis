get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import exp,sin,linspace,sign,abs
from matplotlib.pyplot import plot,grid,xlabel,ylabel

def fun(x):
    #f = x**2 - 4*x*np.sin(x) + (2*np.sin(x))**2
    f = exp(x) - sin(x)
    return f

x=linspace(-4,-2,100)
f=fun(x)
plot(x,f,'r-')
grid(True)
xlabel('x')
ylabel('f')

# Initial interval [a,b]
a, b = -4, -2

M=100         # Maximum number of iterations
eps=1.0e-4   # Tolerance on the interval
delta=1.0e-4 # Tolerance on the function

fa = fun(a)
fb = fun(b)
sa = sign(fa)
sb = sign(fb)

for i in range(M):
    e = b-a
    c = a + 0.5*e
    if abs(e) < eps*abs(c):
        print "Interval size is below tolerance\n"
        break
    fc = fun(c)
    if abs(fc) < delta:
        print "Function value is below tolerance\n"
        break
    sc = sign(fc)
    if sa != sc:
        b = c
        fb= fc
        sb= sc
    else:
        a = c
        fa= fc
        sa= sc
    print "{0:5d} {1:16.8e} {2:16.8e} {3:16.8e}".format(i+1,c,abs(b-a),abs(fc))


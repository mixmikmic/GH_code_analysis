get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import exp,sin,linspace,sign,abs
from matplotlib.pyplot import plot,grid,xlabel,ylabel

def f1(x):
    f = exp(x) - sin(x)
    return f

def f2(x):
    f = x**2 - 4.0*x*sin(x) + (2.0*sin(x))**2
    return f

def f3(x):
    f = x**2 - 4.0*x*sin(x) + (2.0*sin(x))**2 - 0.5
    return f

def bisect(fun,a,b,M=100,eps=1.0e-4,delta=1.0e-4,debug=False):
    fa = fun(a)
    fb = fun(b)
    sa = sign(fa)
    sb = sign(fb)
    
    if abs(fa) < delta:
        return (a,0)

    if abs(fb) < delta:
        return (b,0)

    # check if interval is correct
    if fa*fb > 0.0:
        if debug:
            print "Interval is not admissible\n"
        return (0,1)

    for i in range(M):
        e = b-a
        c = a + 0.5*e
        if abs(e) < eps*abs(c):
            if debug:
                print "Interval size is below tolerance\n"
            return (c,0)
        fc = fun(c)
        if abs(fc) < delta:
            if debug:
                print "Function value is below tolerance\n"
            return (c,0)
        sc = sign(fc)
        if sa != sc:
            b = c
            fb= fc
            sb= sc
        else:
            a = c
            fa= fc
            sa= sc
        if debug:
            print "{0:5d} {1:16.8e} {2:16.8e} {3:16.8e}".format(i+1,c,abs(b-a),abs(fc))
        
    # If we reached here, then there is no convergence
    print "No convergence in %d iterations !!!" % M
    return (0,2)

M=100         # Maximum number of iterations
eps=1.0e-4   # Tolerance on the interval
delta=1.0e-4 # Tolerance on the function
a, b = -4.0, -2.0
r,status = bisect(f1,a,b,M,eps,delta,True)

M=100         # Maximum number of iterations
eps=1.0e-4   # Tolerance on the interval
delta=1.0e-4 # Tolerance on the function
a, b = -4.0, -2.0
r,status = bisect(f2,a,b,M,eps,delta,True)

x=linspace(-3,3,500)
y=f2(x)
plot(x,y)
grid(True)

M=100         # Maximum number of iterations
eps=1.0e-4   # Tolerance on the interval
delta=1.0e-4 # Tolerance on the function
a, b = -3.0, +2.0
r,status = bisect(f3,a,b,M,eps,delta,True)


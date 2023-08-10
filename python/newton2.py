get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import sin,cos,linspace,zeros,abs
from matplotlib.pyplot import plot,xlabel,ylabel,grid

def f(x):
    return (x-1.0)**2 * sin(x)

def df(x):
    return 2.0*(x-1.0)*sin(x) + (x-1.0)**2 * cos(x)

x = linspace(0.0,2.0,100)
plot(x,f(x))
xlabel('x')
ylabel('f(x)')
grid(True)

def newton(x0,m):
    n = 50
    x = zeros(50)
    x[0] = x0
    print "%6d %24.14e" % (0,x[0])
    for i in range(1,50):
        x[i] = x[i-1] - m*f(x[i-1])/df(x[i-1])
        if i > 1:
            r = (x[i] - x[i-1])/(x[i-1]-x[i-2])
        else:
            r = 0.0
        print "%6d %24.14e %14.6e" % (i,x[i],r)
        if abs(f(x[i])) < 1.0e-16:
            break

newton(2.0,1)

newton(2.0,2)


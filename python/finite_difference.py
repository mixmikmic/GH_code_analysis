get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import sin,arange,zeros,pi,abs
from matplotlib.pyplot import loglog,xlabel,ylabel

def f(x):
    return sin(x)

h = 10.0**arange(-1,-15,-1)
df= zeros(len(h))
x = 2.0*pi
f0= f(x)
for i in range(len(h)):
    f1 = f(x+h[i])
    df[i] = (f1 - f0)/h[i]
loglog(h,abs(df-1.0),'o-')
xlabel('h')
ylabel('Error in derivative');


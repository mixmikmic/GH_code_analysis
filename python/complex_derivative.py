get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import sin,arange,zeros,pi,abs,imag
from matplotlib.pyplot import loglog,xlabel,ylabel

def f(x):
    return sin(x)

h = 10.0**arange(-1,-15,-1)
df= zeros(len(h))
x = 2.0*pi
for i in range(len(h)):
    df[i] = imag(f(x+1j*h[i]))/h[i]
loglog(h,abs(df-1.0),'o-')
xlabel('h')
ylabel('Error in derivative')
print df-1.0


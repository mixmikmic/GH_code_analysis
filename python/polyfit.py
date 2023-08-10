get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import linspace,cos,pi,polyfit,polyval
import matplotlib.pyplot as plt

xmin, xmax = 0.0, 1.0
f = lambda x: cos(4*pi*x)

N = 8 # degree, we need N+1 points
x = linspace(xmin, xmax, N+1)
y = f(x)

p = polyfit(x,y,N)

M = 100
xe = linspace(xmin, xmax, M)
ye = f(xe) # exact function
yp = polyval(p,xe)

plt.plot(x,y,'o',xe,ye,'--',xe,yp,'-')
plt.legend(('Data points','Exact function','Polynomial'))
plt.title('Degree '+str(N)+' interpolation');


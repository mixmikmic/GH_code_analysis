dx = 1.
x = 1.
while(dx > 1.e-10):
    dy = (x+dx)*(x+dx)-x*x
    d = dy / dx
    print("%6.0e %20.16f %20.16f" % (dx, d, d-2.))
    dx = dx / 10.
    

((1.+0.0001)*(1+0.0001)-1)

dx = 1.
x = 1.
while(dx > 1.e-10):
    dy = (x+dx)*(x+dx)-x*x
    d = dy / dx
    print("%6.0e %20.16f %20.16f" % (dx, d, d-2.))
    dx = dx / 2.

from math import sin, sqrt, pi
dx = 1.
while(dx > 1.e-10):
    x = pi/4.
    d1 = sin(x+dx) - sin(x); #forward
    d2 = sin(x+dx*0.5) - sin(x-dx*0.5); # midpoint
    d1 = d1 / dx;
    d2 = d2 / dx;
    print("%6.0e %20.16f %20.16f %20.16f %20.16f" % (dx, d1, d1-sqrt(2.)/2., d2, d2-sqrt(2.)/2.) )
    dx = dx / 2.

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot

y = lambda x: x*x

x1 = np.arange(0,10,1)
x2 = np.arange(0,10,0.1)

y1 = np.gradient(y(x1), 1.)
print y1

pyplot.plot(x1,np.gradient(y(x1),1.),'r--o');
pyplot.plot(x1[:x1.size-1],np.diff(y(x1))/np.diff(x1),'b--x');  

pyplot.plot(x2,np.gradient(y(x2),0.1),'b--o');

from scipy.misc import derivative

y = lambda x: x**2

dx = 1.
x = 1.

while(dx > 1.e-10):
    d = derivative(f, x, dx, n=1, order=3)
    print("%6.0e %20.16f %20.16f" % (dx, d, d-2.))
    dx = dx / 10.

from decimal import Decimal

dx = Decimal("1.")
while(dx >= Decimal("1.e-10")):
    x = Decimal("1.")
    dy = (x+dx)*(x+dx)-x*x
    d = dy / dx
    print("%6.0e %20.16f %20.16f" % (dx, d, d-Decimal("2.")))
    dx = dx / Decimal("10.")


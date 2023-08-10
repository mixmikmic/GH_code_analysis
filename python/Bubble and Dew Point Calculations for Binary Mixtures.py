import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

A = 'acetone'
B = 'ethanol'

P = 760
xa = 0.4
xb = 1 - xa

# Antoine's equations. T [deg C], P [mmHg]

Psat = dict()
Psat['acetone'] = lambda T: 10**(7.02447 - 1161.0/(224 + T))
Psat['ethanol'] = lambda T: 10**(8.04494 - 1554.3/(222.65 + T))

T = 65

Ka = Psat[A](T)/P
Kb = Psat[B](T)/P

ya = Ka*xa
yb = Kb*xb

print(ya + yb)

from scipy.optimize import brentq

brentq(lambda T: xa*Psat[A](T)/P + xb*Psat[B](T)/P - 1.0 ,0,100)

def Tbub(X) :
    xa,xb = X
    return brentq(lambda T: xa*Psat[A](T)/P + xb*Psat[B](T)/P - 1.0 ,0,100)

print("Bubble point temperature = {:6.3f} [deg C]".format(Tbub((xa,xb))))

ya = xa*Psat[A](Tbub((xa,xb)))/P
yb = xb*Psat[B](Tbub((xa,xb)))/P

print("Bubble point composition = {:.3f}, {:.3f}".format(ya,yb))

x = np.linspace(0,1)
plt.plot(x,[Tbub((xa,xb)) for (xa,xb) in zip(x,1-x)])
plt.xlabel('x [mole fraction {:s}]'.format(A))
plt.ylabel('Temperature [deg C]')
plt.title('Bubble Point for {:s}/{:s} at {:5.1f} [mmHg]'.format(A,B,P))
plt.grid();

def Tdew(Y):
    ya,yb = Y
    return brentq(lambda T:ya*P/Psat[A](T) + yb*P/Psat[B](T) - 1.0,0,100)

print("Dew point temperature = {:6.3f} [deg C]".format(Tdew((ya,yb))))

xa = ya*P/Psat[A](Tdew((ya,yb)))
xb = yb*P/Psat[B](Tdew((ya,yb)))

print("Dew point composition = {:.3f}, {:.3f}".format(xa,xb))

y = np.linspace(0,1)
plt.plot(y,[Tdew((ya,yb)) for (ya,yb) in zip(y,1-y)])
plt.xlabel('y [mole fraction {:s}]'.format(A))
plt.ylabel('Temperature [deg C]')
plt.title('Dew Point for {:s}/{:s} at {:5.1f} [mmHg]'.format(A,B,P))
plt.grid();




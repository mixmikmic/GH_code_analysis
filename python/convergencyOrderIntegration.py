import numpy as np
from scipy.integrate import odeint
from ddeint import ddeint
import matplotlib.pyplot as plt

def f1(x, t):
    return x

x0 = 1
tf = 10
sol_ref = np.exp(tf)

h = 0.1
for i in range(8):
    t = np.arange(0, tf + h / 2, h)
    x = odeint(f1, x0, t, hmax= h)
    if i == 0:
        auxErr1 = abs( x[-1][0] - sol_ref)
        print("h=%9.2e"% (h), ",  error=%9.2e"% (auxErr1))
    else:
        auxErr2 = abs( x[-1][0] - sol_ref)
        p = np.log(auxErr1 / auxErr2) / np.log(2.0)
        print("h=%9.2e"% (h), ",  error=%9.2e"% (auxErr2), ", p=%9.2e"% (p))
        auxErr1 = auxErr2
    h = h / 2

def f2(x, t):
    return -x(t - np.pi / 2)
hist = lambda t: np.sin(t)

h = np.pi / 2
sol_ref = np.sin(tf)
tf = 10.0 * h

for i in range(8):
    t = np.arange(0, tf + h / 2, h)
    x = ddeint(f2, hist, t)
    if i == 0:
        auxErr1 = abs( x[-1] - sol_ref)
        print("h=%9.2e"% (h), ",  error=%9.2e"% (auxErr1))
    else:
        auxErr2 = abs( x[-1] - sol_ref)
        p = np.log(auxErr1 / auxErr2) / np.log(2.0)
        print("h=%9.2e"% (h), ",  error=%9.2e"% (auxErr2), ", p=%9.2e"% (p))
        auxErr1 = auxErr2
    h = h / 2
    






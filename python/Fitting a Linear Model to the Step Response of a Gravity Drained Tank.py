get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

Cv  = 0.1     # Outlet valve constant [cubic meters/min/meter^1/2]
A   = 1.0     # Tank area [meter^2]

# inlet flow rate in cubic meters/min
def qin(t):
    return 0.15

def deriv(h,t):
    return qin(t)/A - Cv*np.sqrt(h)/A

IC = [0.0]
t = np.linspace(0,200,101)
h = odeint(deriv,IC,t)

plt.plot(t,h)

plt.xlabel('Time [min]')
plt.ylabel('Level [m]')
plt.grid();

q0,h0 = 0,0            # initial conditions
qss = qin(t[-1])       # final input
hss = h[-1]            # python way to get the last element in a list

K = (hss-h0)/(qss-q0)  # step change in output divided by step change in input
print('Steady-State Gain is approximately = ', K)

k = sum(t<25)        # find index in t corresponding to 25 minutes
tk = t[k]
hk = h[k]

tau = -tk/np.log((hss-hk)/(hss-h0))
print('Estimated time constant is ', tau)

u0 = q0
uss = qss

xss = K*(uss - u0)
xpred = xss - xss*np.exp(-t/tau)

plt.plot(t,h)
plt.plot(t,xpred)
plt.legend(['Nonlinear Simulation','Linear Approximation'])
plt.xlabel('Time [min]')
plt.ylabel('Level [m]')
plt.title('Nonlinear Simulation vs. Linear Approximation');




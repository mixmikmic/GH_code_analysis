import numpy
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
from controllers import PIController
get_ipython().magic('matplotlib inline')

# This is the 1,1 element of a Wood and Berry column (see eq 16-12)
K = 12.8
tau = 16.7
theta = 1

ts = numpy.linspace(0, 2*tau, 500)
dt = ts[1]
r = 1

def response(Kc, tau_i):
    Gc = PIController(Kc, tau_i, bias=0)
    y = 0
    ys = numpy.zeros_like(ts)

    for i, t in enumerate(ts):
        e = r - numpy.interp(t - theta, ts, ys)
        
        Gc.change_input(t, e)
        u = Gc.output

        dydt = -1/tau*y + K/tau*u

        # integrate
        y += dydt*dt
        Gc.change_state(Gc.x + Gc.derivative*dt)

        ys[i] = y
    return ys

for Kc in [0.5, 1, 2]:
    plt.plot(ts, response(Kc, 10), label='$K_c={}$'.format(Kc))
plt.axhline(r, label='$y_{sp}$')
plt.legend()

def iae(parameters):
    return scipy.integrate.trapz(numpy.abs(response(*parameters) - r), ts)

def ise(parameters):
    return scipy.integrate.trapz((response(*parameters) - r)**2, ts)

def itae(parameters):
    return scipy.integrate.trapz(numpy.abs(response(*parameters) - r)*ts, ts)

errfuns = [iae, ise, itae]

get_ipython().run_cell_magic('time', '', "optimal_parameters = {}\nfor error in errfuns:\n    name = error.__name__.upper()\n    optimal_parameters[name] = scipy.optimize.minimize(error, [1, 10]).x\n    print(name, *optimal_parameters[name])\n    plt.plot(ts, response(*optimal_parameters[name]), label=name)\nplt.axhline(1, label='setpoint')\nplt.legend(loc='best')")

A, B = 0.586, -0.916
Kc = A*(theta/tau)**B/K
A, B = 1.03, -0.165
tau_i = tau/(A + B*(theta/tau))

print(Kc, tau_i)

plt.plot(ts, response(Kc, tau_i))


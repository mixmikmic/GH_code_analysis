# Numerical integration in Brian2

from brian2 import *

eqs = Equations('dv/dt = -v/tau : volt')
print euler(eqs)

print StateUpdateMethod.stateupdaters.keys()

get_ipython().magic('pinfo NeuronGroup')

linear_eq = Equations('dv/dt = -v / tau : volt')
nonlinear_eq = Equations('''dv/dt = 0.04*v**2 + 5*v + 140 - u + I : 1
                            du/dt = a*(b*v - u)                   : 1''')
additive_noise = Equations('dv/dt = -v / (10*ms) + xi/(10*ms)**.5 : 1')
multiplicative_noise = Equations('dv/dt = -v / (10*ms) + v*xi/(10*ms)**.5 : 1')

# Linear equation
for name, updater in StateUpdateMethod.stateupdaters.items():
    print name, updater.can_integrate(linear_eq, {})
linear_eq

# Non-Linear equation
for name, updater in StateUpdateMethod.stateupdaters.items():
    print name, updater.can_integrate(nonlinear_eq, {})
nonlinear_eq

# Equation with additive noise
for name, updater in StateUpdateMethod.stateupdaters.items():
    print name, updater.can_integrate(additive_noise, {})
additive_noise

for name, updater in StateUpdateMethod.stateupdaters.items():
    print name, updater.can_integrate(multiplicative_noise, {})
multiplicative_noise

print rk4

from IPython.display import Image
Image('http://upload.wikimedia.org/wikipedia/commons/5/52/Heun%27s_Method_Diagram.jpg')

heun = ExplicitStateUpdater('''tangent = f(x,t)
                               x_new = x+0.5*dt*(tangent+f(x+dt*tangent, t+dt))''')
print heun(nonlinear_eq)

# by directly passing the object, state updaters that Brian doesn't know about can be used
G = NeuronGroup(1, nonlinear_eq, method=heun)

StateUpdateMethod.register('heun', heun)  # register the method
G = NeuronGroup(1, nonlinear_eq, method='heun') # now it can be used like any other method


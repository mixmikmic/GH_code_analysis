import sympy
sympy.init_printing()
get_ipython().magic('matplotlib inline')

s = sympy.Symbol('s')
t = sympy.Symbol('t', real=True)
tau = sympy.Symbol('tau', real=True, positive=True)

G = 1/(tau*s + 1)

g = sympy.inverse_laplace_transform(G.subs({tau: 1}), s, t)
g

sympy.plot(g, (t, -1, 10))

stepresponse = sympy.inverse_laplace_transform(G.subs({tau: 1})/s, s, t)
stepresponse

sympy.plot(stepresponse, (t, -1, 10), ylim=(0, 1.1))

sympy.Heaviside(t - tau)

sympy.integrate(g.subs({t: tau})*sympy.Heaviside(t - tau), (tau, -sympy.oo, sympy.oo))

import numpy
import matplotlib.pyplot as plt

tt = numpy.linspace(0, 10, 100)
dt = tt[1]  # space between timesteps 

gt = numpy.exp(-tt)

ut = numpy.ones_like(tt)

plt.plot(tt, ut, tt, gt)
plt.ylim(ymax=1.1)

full_convolution = numpy.convolve(gt, ut)

plt.plot(full_convolution)

yt = full_convolution[:len(tt)]*dt

plt.plot(tt, yt)


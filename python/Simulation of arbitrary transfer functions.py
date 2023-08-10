import numpy
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

K = 1
tau = 5

def u(t):
    if t<1:
        return 0
    else:
        return 1

ts = numpy.linspace(0, 10, 1000)
dt = ts[1]
y = 0
ys = []
for t in ts:
    dydt = -1/tau*y + 1/tau*u(t)
    
    y += dydt*dt
    ys.append(y)

plt.plot(ts, ys)

import scipy.signal

G = scipy.signal.ltisys.TransferFunction(K, [tau, 1])

_, ys_step = G.step(T=ts)
plt.plot(ts, ys_step);

us = [u(t) for t in ts]  # evaluate the input function at all the times
_, ys_lsim, xs = scipy.signal.lsim(G, U=us, T=ts)
plt.plot(ts, ys_lsim, ts, ys);

x = numpy.matrix(numpy.zeros(G.A.shape[0]))
ys_statespace = []
for t in ts:
    xdot = G.A.dot(x) + G.B.dot(u(t))
    y = G.C.dot(x) +  G.D.dot(u(t))
    
    x += xdot*dt
    ys_statespace.append(y[0,0])

plt.plot(ts, ys,
         ts, ys_statespace);




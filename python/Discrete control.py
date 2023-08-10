from __future__ import print_function, division

import numpy
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

K = 3
tau = 2

Kc = 2

ts = numpy.linspace(0, 5, 1000)
dt = ts[1]

y_continuous = []
u_continuous = []
y = 0
sp = 1
for t in ts:
    e = sp - y
    u = Kc*e
    dydt = 1/tau*(K*u - y)
    
    y += dydt*dt
    
    u_continuous.append(u)
    y_continuous.append(y)

plt.subplot(2, 1, 1)
plt.plot(ts, u_continuous)
plt.subplot(2, 1, 2)
plt.plot(ts, y_continuous)

DeltaT = 0.5  # sampling time

u_discrete = []
y_discrete = []
y = 0
sp = 1
next_sample = 0
for t in ts:
    if t >= next_sample:
        e = sp - y
        u = Kc*e
        next_sample += DeltaT
    dydt = 1/tau*(K*u - y)
    y += dydt*dt
    
    u_discrete.append(u)
    y_discrete.append(y)

plt.subplot(2, 1, 1)
plt.plot(ts, u_continuous,
         ts, u_discrete)
plt.subplot(2, 1, 2)
plt.plot(ts, y_continuous,
         ts, y_discrete)

import sympy
sympy.init_printing()

s = sympy.Symbol('s')
t = sympy.Symbol('t', positive=True)

Gc = Kc  # controller
G = K/(tau*s + 1)  # system

G_cl = Gc*G/(1 + Gc*G)

rs = 1/s  # step input r(s)

ys = rs*G_cl  # system output y(s)
es = rs - ys  # error
us = Gc*es  # controller output

yt = sympy.inverse_laplace_transform(ys, s, t)
ut = sympy.inverse_laplace_transform(us, s, t)

sympy.plot(ut, (t, 0, 5))
sympy.plot(yt, (t, 0, 5))

z, q = sympy.symbols('z, q')

rz = 1/(1 - z**-1)

rz.subs(z, q**-1).series()

def sampledvalues(fz, N):
    return sympy.poly(fz.subs(z, q**-1).series(q, 0, N).removeO(), q).all_coeffs()[::-1]

def plotdiscrete(fz, N):
    values = sampledvalues(fz, N)
    times = [n*DeltaT for n in range(N)]
    plt.stem(times, values)

plotdiscrete(rz, 10)

Gcz = Kc

a = 1/tau
b = sympy.exp(-a*DeltaT)
Fz = K*(1 - b)*z**-1/((1 - z**-1)*(1 - b*z**-1))  # In the datasheet table, this corresponds with (1 - e^{-at})
HGz = Fz - z**-1*Fz

plotdiscrete(rz*HGz, 10)
plt.plot(ts, K*(1 - numpy.exp(-ts/tau)))

yz = rz*Gcz*HGz/(1 + Gcz*HGz)

plt.plot(ts, y_discrete)
plotdiscrete(yz, 10)
plt.legend(['Numeric simulation', 'Analytical at sampling points'])

ez = rz - yz
uz = Gcz*ez
plotdiscrete(uz, 10)
plt.plot(ts, u_discrete)

Hs = 1/s*(1 - sympy.exp(-DeltaT*s))

u_single_pulse = 2*Hs
y_single_pulse = sympy.inverse_laplace_transform(G*u_single_pulse, s, t)

sympy.plot(y_single_pulse, (t, 0, 5))

uhs = sum(ai*Hs*sympy.exp(-i*DeltaT*s) 
          for i, ai in enumerate(sampledvalues(uz, 10)))

uhs = 0
a = sampledvalues(uz, 10)
for i in range(10):
    uhs += a[i]*Hs*sympy.exp(-i*DeltaT*s)

ys = uhs*G

yt = sympy.inverse_laplace_transform(ys, s, t)

plt.plot(ts, y_discrete)
plt.plot(ts, [sympy.N(yt.subs(t, ti)) for ti in ts], '--')


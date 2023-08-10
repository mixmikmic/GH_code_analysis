get_ipython().magic('matplotlib inline')
import numpy as np
from kolzur_filter import kz_filter, kzft, kzp, _kz_coeffs
import matplotlib.pyplot as plt

plt.plot(_kz_coeffs(101, 2), label='k=2')
plt.plot(_kz_coeffs(101, 3), label='k=3')
plt.plot(_kz_coeffs(101, 5), label='k=5')
plt.xlabel('Index')
plt.ylabel('Coefficient')
plt.legend()

dt = 0.1
t = np.arange(0, 200+dt, dt)
x = np.sin(2*np.pi*0.05*t)+0.1*np.sin(2*np.pi*0.25*t)

plt.plot(t, x)
plt.xlabel('Time')
plt.ylabel('Signal')

plt.plot(t, x, label='Original')

k = 3
for m in [21, 51, 101]:
    w = int(k*(m-1)/2)
    plt.plot(t[w:-w], kz_filter(x, m, k), label='m={}'.format(m))

plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()

plt.plot(t, x, label='True signal')

k = 3
for m in [5, 9]:
    xkzftf = 2*np.real(np.sum(kzft(x, [0.05, 0.25], m, k, dt=dt), axis=0))
    w = int((t.size-xkzftf.size)/2)
    plt.plot(t[w:-w], xkzftf, label='m={}'.format(m))

plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()

k = 3
nu = np.linspace(0., 0.5, num=500)
for m in [15, 30, 45]:
    plt.plot(nu, kzp(x, nu, m, k, dt=dt), label='m={}'.format(m))

plt.xlabel('Frequency')
plt.ylabel('Perdiodogram')
plt.xlim(0, 0.5)
plt.legend()

# Determine random points to remove
idx = np.random.permutation(t.size)[:int(t.size*0.5)]

xm = x.copy()
xm[idx] = np.nan

plt.plot(t, xm)
plt.xlabel('Time')
plt.ylabel('Signal')

plt.plot(t, x, label='True signal')

k = 3
m = 20

xkzftf = 2*np.real(np.sum(kzft(xm, [0.05, 0.25], m, k, dt=dt), axis=0))
w = int((t.size-xkzftf.size)/2)
plt.plot(t[w:-w], xkzftf, label='KZFT m={}'.format(m))
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()

nu = np.linspace(0., 0.5, num=500)
plt.plot(nu, kzp(x, nu, m, k, dt=dt), label='True signal')
plt.plot(nu, kzp(xm, nu, m, k, dt=dt), label='Sparse signal')
plt.xlabel('Frequency')
plt.ylabel('Perdiodogram')
plt.xlim(0, 0.5)
plt.legend()

xn = x.copy()+np.random.normal(scale=1., size=t.size)
plt.plot(t, xn)

plt.plot(t, x, label='True signal')

k = 3
m = 20
xkzftf = 2*np.real(np.sum(kzft(xn, [0.05, 0.25], m, k, dt=dt), axis=0))
w = int((t.size-xkzftf.size)/2)
plt.plot(t[w:-w], xkzftf, label='KZFT m={}'.format(m))

plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()

nu = np.linspace(0., 0.5, num=500)
plt.plot(nu, kzp(x, nu, m, k, dt=dt), label='True signal')
plt.plot(nu, kzp(xn, nu, m, k, dt=dt), label='Noisy signal')
plt.xlabel('Frequency')
plt.ylabel('Perdiodogram')
plt.xlim(0, 0.5)
plt.legend()

xnm = np.copy(xn)
xnm[idx] = np.nan
plt.plot(t, xnm, ls='', marker='.')

plt.plot(t, x, label='True signal')

k = 3
m = 20
xkzftf = 2*np.real(np.sum(kzft(xnm, [0.05, 0.25], m, k, dt=dt), axis=0))
w = int((t.size-xkzftf.size)/2)
plt.plot(t[w:-w], xkzftf, label='KZFT m={}'.format(m))

plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()

nu = np.linspace(0., 0.5, num=500)
plt.plot(nu, kzp(x, nu, m, k, dt=dt), label='True signal')
plt.plot(nu, kzp(xnm, nu, m, k, dt=dt), label='Sparse and noisy signal')
plt.xlabel('Frequency')
plt.ylabel('Perdiodogram')
plt.xlim(0, 0.5)
plt.legend()


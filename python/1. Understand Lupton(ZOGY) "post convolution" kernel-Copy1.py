import sympy as sym
import sympy.stats as symstat
from sympy.interactive import printing
sym.init_printing()

x,F,m,s,k,n = sym.symbols("x F m s k n")
G = 1/(s*sym.sqrt(2*sym.pi)) * sym.exp(-(x-m)**2/(2*s**2))
G

G.evalf(subs={x: 0, m: 0, s: 1})

import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

qq = [G.evalf(subs={x: xx, m: 0, s: 1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)

H = sym.fourier_transform(G, x, k)
H

qq = [H.evalf(subs={k: xx, m: 0, s: 1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)

D = sym.KroneckerDelta(x, 3.0)
xrange = np.arange(-4,4,0.01)
xrange[np.abs(xrange-3.0) <= 1e-5] = 3.0
qq = [D.evalf(subs={x: xx}) for xx in xrange]
plt.plot(xrange, qq)

from scipy.fftpack import fft, fftfreq
t = np.linspace(0., 100., 2000)
signal = np.sin(t / 2.)
plt.plot(t, signal)

npts = len(t)
FFT = fft(signal)
freqs = fftfreq(npts, t[1]-t[0])
plt.plot(freqs, abs(FFT))

from scipy.fftpack import ifft
signal2 = ifft(FFT)
plt.plot(t, signal2.real)

# Define kappa(x) (the spatial matching kernel) as a simple narrow Gaussian:
sigk = sym.symbols("sigk")

kappa = 1/(sigk*sym.sqrt(2*sym.pi))*sym.exp(-(x-m)**2/(2*sigk**2))
qq = [kappa.evalf(subs={x: xx, m: 0, sigk: 1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)
kappa

# Now define kappa_ft(k) (the FT of the matching kernel kappa):
kappa_ft = sym.fourier_transform(kappa, x, k)
qq = [kappa_ft.evalf(subs={k: xx, m: 0, sigk: 1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)
plt.xlim(-3, 3)
kappa_ft

sig1, sig2 = sym.symbols("sig1 sig2")
# post_conv_kernel = sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
post_conv_kernel_ft = sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft**2))
qq = [post_conv_kernel_ft.evalf(subs={k:xx, m:0, sigk:0.1, sig1:1, sig2:1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)
post_conv_kernel_ft

# This takes a long time so let's look at it numerically instead...
if False:
    post_conv_kernel = sym.inverse_fourier_transform(post_conv_kernel_ft, k, x, noconds=False)
    post_conv_kernel

def kappa(x, m=0., sigk=1.0):
    out = 1/(sigk*np.sqrt(2*np.pi))*np.exp(-(x-m)**2./(2.*sigk**2.))
    return out / out.sum() / (x[1] - x[0])

x = np.arange(-10,10,0.01)
plt.plot(x[abs(x)<3], kappa(x)[abs(x)<3])

from scipy.fftpack import fft, fftfreq

def kappa_ft(x, m=0., sigk=1.0):
    x = np.arange(-10,10,1)
    kp = kappa(x, m, sigk)
    npts = len(x)
    FFT = fft(kp)
    freqs = fftfreq(npts, x[1]-x[0])  # assumes uniformly sampled x!
    return FFT, freqs

FFT, freqs = kappa_ft(x)
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(abs(FFT)))
plt.xlim(-3, 3)

# post_conv_kernel = sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
def post_conv_kernel_ft(x, sig1=1., sig2=1., m=0., sigk=1.):
    x = np.arange(-10,10,1)
    kft, freqs = kappa_ft(x, m, sigk)
    return np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2)), freqs

kft, freqs = post_conv_kernel_ft(x)
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(kft.real))
plt.xlim(-3, 3)

from scipy.fftpack import ifft

def post_conv_kernel(x, sig1=1., sig2=1., m=0., sigk=1.0):
    x = np.arange(-10,10,1)
    kft, freqs = post_conv_kernel_ft(x, sig1, sig2, m, sigk)
    out = ifft(kft)
    npts = len(x)
    freqs = fftfreq(npts, x[1]-x[0])  # assumes uniformly sampled x!
    return out, freqs

pck, freqs = post_conv_kernel(x)
xx = np.fft.ifftshift(freqs)
plt.plot(xx, pck.real)
plt.xlim(-3, 3)

# Ensure there is no trend in the imag. component:
plt.plot(np.fft.ifftshift(freqs), pck.imag)

pck = post_conv_kernel(x, sig2=0.)
xx = np.fft.ifftshift(x)
plt.plot(xx[np.abs(xx) <= 0.5], pck.real[np.abs(xx) <= 0.5])

pck = post_conv_kernel(x, sigk=0.01)
xx = np.fft.ifftshift(x)
plt.plot(xx[np.abs(xx) <= 0.5], pck.real[np.abs(xx) <= 0.5])

pck = post_conv_kernel(x, sig1=0.1)
xx = np.fft.ifftshift(x)
plt.plot(xx[np.abs(xx) <= 0.5], pck.real[np.abs(xx) <= 0.5])

pck = post_conv_kernel(x, m=3.) ## look at offset
xx = np.fft.ifftshift(x)
plt.plot(xx[np.abs(xx) <= 0.5], pck.real[np.abs(xx) <= 0.5])




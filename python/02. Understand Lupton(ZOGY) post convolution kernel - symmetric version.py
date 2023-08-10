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

from scipy.fftpack import fft, fftfreq
t = np.linspace(0., 100., 2000)
signal = np.sin(t / 2.)
#plt.plot(t, signal)

npts = len(t)
FFT = fft(signal)
freqs = fftfreq(npts, t[1]-t[0])
#plt.plot(freqs, np.abs(FFT))
#plt.xlim(-3, 3)

from scipy.fftpack import ifft
signal2 = ifft(FFT)
#plt.plot(t, signal2.real)

# Define phi1(x) and phi2(x) (the spatial psfs of image 1 and 2) as simple Gaussians (with different sigmas):
sig1, sig2 = sym.symbols("sig1 sig2")

phi = 1/(s*sym.sqrt(2*sym.pi))*sym.exp(-(x-m)**2/(2*s**2))
phi1 = phi.subs(s, sig1)
phi2 = phi.subs(s, sig2)
qq = [phi1.evalf(subs={x: xx, m: 0, sig1: 1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)
phi1

# Now define kappa_ft(k) (the FT of the matching kernel kappa):
phi1_ft = sym.fourier_transform(phi1, x, k)
phi2_ft = sym.fourier_transform(phi2, x, k)
qq = [phi1_ft.evalf(subs={k: xx, m: 0, sig1: 1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)
phi1_ft

# Use sd1, sd2 for the sigma_1 and sigma_2 (yes, it's confusing!)
# This is unstable (zero denominator) for small values of phi, so we add one to help out.
sd1, sd2 = sym.symbols("sd1 sd2")
post_conv_kernel_ft = sym.sqrt((sd1**2 + sd2**2) / (1. + sd1**2 * phi2_ft**2 + sd2**2 * phi1_ft**2))
qq = [post_conv_kernel_ft.evalf(subs={k:xx, m:0, sig1:1, sig2:1, sd1:1, sd2:1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)
post_conv_kernel_ft

# This takes a long time so let's look at it numerically instead...
if False:
    post_conv_kernel = sym.inverse_fourier_transform(post_conv_kernel_ft, k, x, noconds=False)
    post_conv_kernel

def phi(x, m=0., sig=1.0):
    out = 1/(sig*np.sqrt(2*np.pi))*np.exp(-(x-m)**2./(2.*sig**2.))
    return out / out.sum() / (x[1] - x[0])

x = np.arange(-6,6,0.01)
plt.plot(x, phi(x, sig=1.0))
plt.plot(x, phi(x, sig=0.6))
plt.xlim(-3, 3)

def phi_ft(x, m=0., sig=1.0):
    kp = phi(x, m, sig)
    npts = len(x)
    FFT = fft(kp)
    FFT *= (x[1]-x[0])
    freqs = fftfreq(npts, x[1]-x[0])  # assumes uniformly sampled x!
    return FFT, freqs

FFT1, freqs1 = phi_ft(x, sig=1.0)
FFT2, freqs2 = phi_ft(x, sig=0.6)
plt.plot(np.fft.fftshift(freqs1), np.fft.fftshift(abs(FFT1)))
plt.plot(np.fft.fftshift(freqs2), np.fft.fftshift(abs(FFT2)))
plt.xlim(-3, 3)

# post_conv_kernel = np.sqrt((sd1**2 + sd2**2) / (sd1**2 * phi2_ft(k)**2 + sd2**2 * phi1_ft(k)**2))
def post_conv_kernel_ft(x, sd1=1., sd2=1., sig1=1., sig2=1., m=0., offset=1.):
    phi1_ft, freqs = phi_ft(x, m, sig1)
    phi2_ft, freqs = phi_ft(x, m, sig2)
    return np.sqrt((sd1**2 + sd2**2) / (sd1**2 * phi2_ft**2 + sd2**2 * phi1_ft**2 + offset)), freqs

kft, freqs = post_conv_kernel_ft(x, sig1=1., sig2=0.6)
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(kft.real))
plt.xlim(-3, 3)

def post_conv_kernel(x, sd1=1., sd2=1., sig1=1., sig2=1., m=0., offset=1.):
    kft, freqs = post_conv_kernel_ft(x, sd1, sd2, sig1, sig2, m, offset)
    out = ifft(kft)
    return out

pck = post_conv_kernel(x,  sig1=1., sig2=0.6)
print pck.real.max() - np.sqrt(2.)
print pck.real.min()
print pck.real.sum()
plt.plot(x, np.fft.ifftshift(pck.real))
plt.ylim(-0.0015, 0.0001)

# Ensure there is no trend in the imag. component:
plt.plot(x, np.fft.ifftshift(pck.imag))

pck = post_conv_kernel(x, sig1=1., sig2=1e-5)
print pck.real.max() - np.sqrt(2.)
print pck.real.min()
print pck.real.sum()
plt.plot(x, np.fft.ifftshift(pck.real))
plt.ylim(-0.00001, 0.0001)

pck = post_conv_kernel(x, sig1=1., sig2=0.9)
print pck.real.max() - np.sqrt(2.)
print pck.real.min()
print pck.real.sum()
plt.plot(x, np.fft.ifftshift(pck.real))
plt.ylim(-0.05, 0.1)

pck = post_conv_kernel(x, sig1=0.1, sig2=1.0)
print pck.real.max() - np.sqrt(2.)
print pck.real.min()
print pck.real.sum()
plt.plot(x, np.fft.ifftshift(pck.real))
plt.ylim(-0.1, 0.1)

pck = post_conv_kernel(x, m=3.) ## look at offset
print pck.real.max() - np.sqrt(2.)
print pck.real.min()
print pck.real.sum()
plt.plot(x, np.fft.ifftshift(pck.real))
plt.ylim(-0.01, 0.01)

pck = post_conv_kernel(x, sig1=1.0, sig2=1.0, offset=0.01)
print pck.real.max() - np.sqrt(2.)
print pck.real.min()
print pck.real.sum()
plt.plot(x, np.fft.ifftshift(pck.real))
plt.ylim(-0.1, 0.1)




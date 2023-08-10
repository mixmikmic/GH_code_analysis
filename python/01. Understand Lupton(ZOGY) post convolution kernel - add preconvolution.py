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
plt.plot(freqs, np.abs(FFT))
plt.xlim(-3, 3)

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
kappa_ft

sig1, sig2 = sym.symbols("sig1 sig2")
# post_conv_kernel = sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
post_conv_kernel_ft = sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft**2))
qq = [post_conv_kernel_ft.evalf(subs={k:xx, m:0, sigk:1, sig1:1, sig2:1}) for xx in np.arange(-5,5,0.1)]
plt.plot(np.arange(-5,5,0.1), qq)
post_conv_kernel_ft

# This takes a long time so let's look at it numerically instead...
if False:
    post_conv_kernel = sym.inverse_fourier_transform(post_conv_kernel_ft, k, x, noconds=False)
    post_conv_kernel

def gaussian(x, m=0., s=1.0):
    out = 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2./(2.*s**2.))
    return out / out.sum() / (x[1] - x[0])

x = np.arange(-5,5,0.1)
plt.plot(x, gaussian(x))
plt.xlim(-3, 3)

def gaussian_ft(x, m=0., s=1.0):
    kp = gaussian(x, m, s)
    npts = len(x)
    FFT = fft(kp)
    FFT *= (x[1]-x[0])
    freqs = fftfreq(npts, x[1]-x[0])  # assumes uniformly sampled x!
    return FFT, freqs

FFT, freqs = gaussian_ft(x, s=1.)
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(abs(FFT)))
plt.xlim(-3, 3)

# post_conv_kernel = sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
def post_conv_kernel_ft(x, sig1=1., sig2=1., m=0., sigk=1.):
    kft, freqs = gaussian_ft(x, m, sigk)
    return np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2)), freqs

kft, freqs = post_conv_kernel_ft(x, sigk=1.)
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(kft.real))
plt.xlim(-3, 3)

def post_conv_kernel(x, sig1=1., sig2=1., m=0., sigk=1.):
    kft, freqs = post_conv_kernel_ft(x, sig1, sig2, m, sigk)
    out = ifft(kft)
    return out

pck = post_conv_kernel(x, sigk=1.)
print pck.real.max() - np.sqrt(2.)
print pck.real.min()
print pck.real.sum()
plt.plot(x, np.fft.ifftshift(pck.real))
plt.ylim(-0.02, 0.01)

# Ensure there is no trend in the imag. component:
plt.plot(x, np.fft.ifftshift(pck.imag))

def post_conv_kernel_ft2(x, sig1=1., sig2=1., m=0., sigk=1., sigM=2., delta=1e-5):
    kft, freqs = gaussian_ft(x, m, sigk)
    mft, freqs = gaussian_ft(x, m, sigM)
    return np.sqrt((sig1**2 + sig2**2 + delta) / (sig1**2 * np.abs(mft)**2 + sig2**2 * np.abs(kft)**2 + delta)), freqs
    #return np.sqrt((sig1**2 + sig2**2 + delta) / (sig1**2 * mft**2 + sig2**2 * kft**2 + delta)), freqs

x = np.arange(-15,15,1)
kft, freqs = post_conv_kernel_ft2(x, sig1=150., sig2=150., sigk=0.5, sigM=2., delta=0.)
print np.abs(kft).real.min(), np.abs(kft).real.max()
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(kft.real))
#plt.xlim(-3, 3)#; plt.ylim(-0.05, 0.05)

pck = ifft(np.fft.fftshift(kft))
print np.abs(pck).real.sum()
plt.plot(x, np.fft.ifftshift(pck).real)

def post_conv_kernel_ft3(x, sig_r=1., sig_n=1., m=0., sigR=1., sigN=1.2):
    P_r, freqs = gaussian_ft(x, m, sigR)
    P_n, freqs = gaussian_ft(x, m, sigN)
    return P_r / np.sqrt(sig_n**2 * np.abs(P_r)**2 + sig_r**2 * np.abs(P_n)**2), freqs

x = np.arange(-127,128,1)
kft, freqs = post_conv_kernel_ft3(x, sig_r=1., sig_n=1., sigR=1., sigN=2.)
print kft.real.min()
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(kft.real))
#plt.xlim(-3, 3)#; plt.ylim(-0.05, 0.05)

pck = ifft(kft)
print pck.real.max(), pck.real.min(), pck.real.sum()
#plt.plot(x, np.fft.ifftshift(pck.real))
plt.plot(x, pck.real)
plt.xlim(-3, 3)
#plt.ylim(-0.02, 0.01)




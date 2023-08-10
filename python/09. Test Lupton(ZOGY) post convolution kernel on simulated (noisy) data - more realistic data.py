import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14.0, 4.0)

np.random.seed(66)
from numpy.polynomial.chebyshev import chebval

from scipy.fftpack import fft, ifft, fftfreq

def gaussian(x, m=0., s=1.0):
    out = 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2./(2.*s**2.))
    return out / out.sum() / (x[1] - x[0])

def gaussian_ft(x, m=0., s=1.0):
    kp = gaussian(x, m, s)
    FFT = fft(kp)
    return FFT

# post_conv_kernel = sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
def post_conv_kernel_ft(x, sig1=1., sig2=1., m=0., sigk=1.):
    kft = gaussian_ft(x, m, sigk)
    return np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))

def post_conv_kernel(x, sig1=1., sig2=1., m=0., sigk=1.):
    kft = post_conv_kernel_ft(x, sig1, sig2, m, sigk)
    out = ifft(kft)
    return out

# post_conv_psf = phi_1(k) * sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
# we'll parameterize phi_1(k) as a gaussian with sigma "psfsig1".
def post_conv_psf_ft(x, sig1=1., sig2=1., m=0., sigk=1., psfsig1=1.):
    kft = gaussian_ft(x, m, sigk)
    sig1ft = gaussian_ft(x, m, psfsig1)
    return sig1ft * np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))

def post_conv_psf(x, sig1=1., sig2=1., m=0., sigk=1., psfsig=1.):
    kft = post_conv_psf_ft(x, sig1, sig2, m, sigk, psfsig)
    out = ifft(kft)
    return out

def chebBasis(x, ord):
    coef = np.zeros(ord+1)
    coef[-1] = 1
    ch = chebval(x, coef)
    return ch, coef

def chebGauss(x, m=0., s=1., ord=0, beta=1.):
    ga = gaussian(x, m, s/beta)
    ch, coef = chebBasis(x, ord)
    #if ord > 0:  # best not to "fix" the basis funcs.
    #    ch -= ch.min()
    #ch /= ch.max()
    print s, ord, coef #, sum(ga), sum(ch), sum(ga*ch)
    out = ga * ch
    return out

if False:
    x = np.arange(-9, 10, 1.)  # This is the axis that we will compute the conv. kernels on
    h0 = chebval(x, [1, 0, 0])
    h1 = chebval(x, [0, 1, 0])
    h2 = chebval(x, [0, 0, 1])/10
    h3 = chebval(x, [0, 0, 0, 1])/100
    plt.plot(x, h0)
    plt.plot(x, h1)
    plt.plot(x, h2)
    plt.plot(x, h3)
    
if False:
    gh0 = gaussian(x) * h0
    gh1 = gaussian(x) * h1
    gh2 = gaussian(x) * h2
    gh3 = gaussian(x) * h3
    plt.plot(x, gh0)
    plt.plot(x, gh1)
    plt.plot(x, gh2)
    plt.plot(x, gh3)

xim = np.arange(-4000, 4000, 1.0)  # assume image coords are centered on sources
n_sources = 50
psf1 = 1.6 # sigma in pixels im1 will be template
psf2 = 2.2 # sigma in pixels im2 will be science image
source_x_t = np.random.uniform(xim.min(), xim.max(), size=n_sources)
source_x_s = source_x_t   # add an offset?
source_flux_t = np.random.uniform(0, 30, size=n_sources)
source_flux_s = source_flux_t.copy()  # fluxes of sources in science im
x_changed = np.argmin(np.abs(source_x_t))
source_flux_s[x_changed] *= 1.5  # make the source closest to x=0 have a small change in flux
print source_x_t[x_changed], source_x_t[x_changed], source_flux_t[x_changed], source_flux_s[x_changed]
xcen = int(source_x_t[x_changed])
im1 = np.zeros(len(xim))
im2 = im1.copy()
for i, cen in enumerate(source_x_t):
    source = gaussian(xim, m=cen, s=psf1)
    im1 += source_flux_t[i] * source / source.sum()
    source = gaussian(xim, m=source_x_s[i], s=psf2)
    im2 += source_flux_s[i] * source / source.sum()
sig1 = 0.2  # sigma of template
sig2 = 0.2  # sigma of science image
im1_noise = np.random.normal(scale=sig1, size=len(im1))
im2_noise = np.random.normal(scale=sig2, size=len(im2))
im1 += im1_noise
im2 += im2_noise
print im1.sum(), im2.sum()
plt.plot(xim, im1); plt.plot(xim, im2); plt.plot(xim, im2-im1)

if False:
    im2_psf = gaussian(x, s=psf2)
    im2_psf /= im2_psf.sum()
    im2_preconv = np.convolve(im2, im2_psf, mode='same')
    print im2.sum(), im2_preconv.sum()
    plt.plot(xim, im1); plt.plot(xim, im2); plt.plot(xim, im2_preconv); plt.plot(xim, im2-im1)

## Don't pre-convolve?
im2_preconv = im2

# Parameters from stack
sigGauss = [0.75, 1.5, 3.0]
degGauss = [4, 2, 2]
betaGauss = 1.   # in the Becker et al. paper sigGauss is 1 and LSST PSF is around 2 pixels?
spatialKernelOrder = 2  # 2  # polynomial for modifying the shape of the kernel across the image
spatialBackgroundOrder = 1  # 1  # polynomial for adding background gradient to fit
# Parameters from and Becker et al. (2012)
#sigGauss = [0.75, 1.5, 3.0]
#degGauss = [6, 4, 2]

x = np.arange(-9, 10, 1.)  # This is the axis that we will compute the conv. kernels on
basis = [chebGauss(x, m=0, s=sig, ord=deg, beta=betaGauss)          for i,sig in enumerate(sigGauss) for deg in range(degGauss[i])] #, kernelOrd=ko) for ko in range(spatialKernelOrder+1)]
basis = np.vstack(basis).T  # put the bases into columns

for b in basis.T:
    #print b
    plt.plot(x, b)

# Single call to do it with all bases
# First use the original (non spatially modified) basis
basis2 = [np.convolve(im1, b, mode='same') for b in basis.T]
basis2 = np.vstack(basis2).T

# Then make the spatially modified basis by simply multiplying the constant
#  basis (basis2 from above) by a polynomial along the image coordinate.
# Note that since we are *not* including i=0, this *does not include* basis2.
if spatialKernelOrder > 0:
    xx = xim/np.max(np.abs(xim))
    basis2m = [b * xx**i for i in range(1, spatialKernelOrder+1) for b in basis2.T]
    basis2m = np.vstack(basis2m).T
    basis2 = np.hstack([basis2, basis2m])

# Then make the spatial background part
if spatialBackgroundOrder >= 0:
    bgBasis = [chebBasis(xim, ord)[0] for ord in range(spatialBackgroundOrder+1)]
    bgBasis = np.vstack(bgBasis).T
    basis2 = np.hstack([basis2, bgBasis])

#%timeit np.linalg.lstsq(basis2, im2_preconv)
pars_old, resid, _, _ = np.linalg.lstsq(basis2, im2_preconv)
print pars_old

#%%timeit 
#basis2 /= basis2.sum(0)
b = np.dot(basis2.T, im2_preconv) #(basis2.T * im2_preconv).sum(1)
M = np.dot(basis2.T, basis2)
pars, resid, _, _ = np.linalg.lstsq(M, b)
print pars

b = np.dot(basis2.T, im2_preconv) #(basis2.T * im2_preconv).sum(1)
M = np.dot(basis2.T, basis2)
pars, resid, _, _ = np.linalg.lstsq(M, b)
print pars
print 'Difference (log10):\n', np.log10(np.abs((pars-pars_old)/pars))

fit = (pars * basis2).sum(1)
print basis2.shape, fit.shape, pars.shape
#plt.plot(x, im2 - fit)  # science - convolved template (red)
#plt.plot(x, im2 - im1)  # science - original template (blue)
plt.rcParams['figure.figsize'] = (12.0, 4.0)
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(xim, im1)  # original template (red)
ax1.plot(xim, fit)  # convolved template (blue)  -- note looks purple because it's right on top of im2
ax1.plot(xim, im2_preconv, ls='-.', lw=3)  # preconvolved science image (dotted, purple)
ax1.plot(xim, im2_preconv - fit)  # diffim (grey)

ax2.plot(xim, im2_preconv - fit)
plt.xlim(xcen-50, xcen+50)
plt.show()

print np.sum((im2_preconv-fit)**2), np.sum((im2-im1)**2)

kfit = (pars[:basis.shape[1]] * basis).sum(1)
print kfit.sum()
kfit /= kfit.sum()
plt.plot(x, kfit)  # this plots the matching kernel

conv_im1 = np.convolve(im1, kfit, mode='same')
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(xim, im1)  # original template (red)
ax1.plot(xim, conv_im1)  # convolved template (blue)
ax1.plot(xim, im2_preconv, ls='-.', lw=3)  # science image (dotted, purple)
ax1.plot(xim, im2_preconv - conv_im1)  # diffim (grey)

ax2.plot(xim, im2_preconv - conv_im1)  # diffim
plt.xlim(xcen-50, xcen+50)
plt.show()

print np.sum((im2_preconv-conv_im1)**2), np.sum((im2_preconv-fit)**2), np.sum((im2-im1)**2)

def kernel_ft(kernel):
    kp = kernel
    FFT = fft(kp)
    return FFT
def post_conv_kernel_ft(kernel, sig1=1., sig2=1.):
    kft = kernel_ft(kernel)
    return np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))
def post_conv_kernel(kernel, sig1=1., sig2=1.):
    kft = post_conv_kernel_ft(kernel, sig1, sig2)
    out = ifft(kft)
    return out

pck = post_conv_kernel(kfit, sig1=sig2, sig2=sig1)
print pck.real.max()
print pck.real.min()
print pck.real.sum()
print np.argmax(pck)
plt.plot(x, np.fft.ifftshift(pck.real))
plt.ylim(np.min(pck.real), np.abs(np.min(pck.real))) #; plt.xlim(-2, 2)

pck = np.fft.ifftshift(pck.real)
print pck.sum(), pck.max(), pck.min()
#pck /= pck.sum()
pci = np.convolve(im2_preconv-conv_im1, pck, mode='same')
plt.plot(xim, pci)  # red - corrected diffim
plt.plot(xim, im2_preconv-conv_im1)  # blue - original diffim

import pandas as pd
df = pd.DataFrame({'corr': pci, 'orig': im2_preconv-conv_im1})
df.plot.hist(alpha=0.5, bins=40)
print 'Corrected:', np.mean(pci), np.std(pci)
print 'Original: ', np.mean(im2_preconv-conv_im1), np.std(im2_preconv-conv_im1)
print 'Expected: ', np.sqrt(sig1**2 + sig2**2)

# post_conv_psf = phi_1(k) * sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
# we'll parameterize phi_1(k) as a gaussian with sigma "psfsig1".
def post_conv_psf_ft(x, kernel, sig1=1., sig2=1., psfsig1=1.):
    kft = kernel_ft(kernel)
    sig1ft = gaussian_ft(x, s=psfsig1)
    out = sig1ft * np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))
    return out
def post_conv_psf(x, kernel, sig1=1., sig2=1., psfsig1=1.):
    kft = post_conv_psf_ft(x, kernel, sig1, sig2, psfsig1)
    out = ifft(kft)
    return out

pcf = post_conv_psf(x, kernel=kfit, sig1=sig2, sig2=sig1, psfsig1=psf2)  # psfsig is sigma of psf of im2 (science image)
pcf = pcf.real / pcf.real.sum()
plt.plot(x, pcf)  # red - corrected PSF
phi2 = gaussian(x, s=psf2) * (x[1]-x[0]) ## compare to phi_1(x)
plt.plot(x, phi2)  # blue - original PSF

tmp1 = np.convolve(pci, pcf, mode='same')
plt.plot(xim, tmp1)  # red - corrected
tmp2 = np.convolve(im2_preconv-conv_im1, phi2, mode='same')
plt.plot(xim, tmp2)  # blue - original

df = pd.DataFrame({'corr': tmp1, 'orig': tmp2})
df.plot.hist(alpha=0.5, bins=50)

print tmp1.std()*5., tmp2.std()*5.
print np.sum(np.abs(tmp1) > tmp1.std()*5.), np.sum(np.abs(tmp2) > tmp2.std()*5.)

import scipy.stats
tmp1a, low, upp = scipy.stats.sigmaclip(tmp1)
tmp2a, low, upp = scipy.stats.sigmaclip(tmp2)
print tmp1a.std()*5., tmp2a.std()*5.

det1 = xim[np.abs(tmp1) > tmp1a.std()*5.]
det2 = xim[np.abs(tmp2) > tmp2a.std()*5.]
print '1:', len(det1)
if len(det1) > 0: 
    print det1.min(), det1.max()
print '2:', len(det2)
if len(det2) > 0:
    print det2.min(), det2.max()

xaxs = np.linspace(df.min()[0], df.max()[0])
plt.plot(xaxs, 200*gaussian(xaxs, s=tmp1a.std()), color='r')
plt.plot(xaxs, 200*gaussian(xaxs, s=tmp2a.std()), color='b')
plt.plot(np.repeat(tmp1a.std()*5., 2), [-0, 800], color='r')
plt.plot(np.repeat(-tmp1a.std()*5., 2), [-0, 800], color='r')
plt.plot(np.repeat(tmp2a.std()*5., 2), [-0, 800], color='b')
plt.plot(np.repeat(-tmp2a.std()*5., 2), [-0, 800], color='b')

plt.plot(xim, tmp1)  # red - corrected
plt.plot(det1, np.repeat(tmp1.max(), len(det1)), '|', color='r')
plt.plot([xim.min(), xim.max()], np.repeat(tmp1a.std()*5., 2), color='r')
plt.plot(xim, tmp2)  # blue - original
plt.plot(det2, np.repeat(tmp2.max(), len(det2)), '|', color='b')
plt.plot([xim.min(), xim.max()], np.repeat(tmp2a.std()*5., 2), color='b')
plt.xlim(xcen-200, xcen+200)




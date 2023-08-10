import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12.0, 4.0)

np.random.seed(66)

from scipy.fftpack import fft, ifft, fftfreq

def gaussian(x, m=0., s=1.0):
    out = 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2./(2.*s**2.))
    return out / out.sum() / (x[1] - x[0])

x = np.arange(-5,5,0.1)

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

xim = np.arange(-400, 400, 0.1)  # make it ~8000 pixels. Assume image coords are centered on sources
psf1 = 0.8  # im1 will be template
psf2 = 1.2  # im2 will be science image
source_x_t = np.array([-2.0, 2.0, -8.0, 8.0])
source_x_s = source_x_t - 0.2   # add an offset?
source_flux_t = np.array([2., 7., 4., 13.]) * 2.
source_flux_s = np.array([3., 7., 4., 13.]) * 2.  # fluxes of sources in science im
im1 = np.zeros(len(xim))
im2 = im1.copy()
for i, cen in enumerate(source_x_t):
    source = gaussian(xim, m=cen, s=psf1)
    im1 += source_flux_t[i] * source / source.sum()
    source = gaussian(xim, m=source_x_s[i], s=psf2)
    im2 += source_flux_s[i] * source / source.sum()
sig1 = 0.02  # variance of template
sig2 = 0.02  # variance of science image
im1_noise = np.random.normal(scale=sig1, size=len(im1))
im2_noise = np.random.normal(scale=sig2, size=len(im2))
im1 += im1_noise
im2 += im2_noise
print im1.sum(), im2.sum()
plt.plot(xim, im1); plt.plot(xim, im2); plt.plot(xim, im2-im1)
plt.xlim(-20, 20)

if False:
    im2_psf = gaussian(x, s=psf2)
    im2_psf /= im2_psf.sum()
    im2_preconv = np.convolve(im2, im2_psf, mode='same')
    print im2.sum(), im2_preconv.sum()
    plt.plot(xim, im1); plt.plot(xim, im2); plt.plot(xim, im2_preconv); plt.plot(xim, im2-im1)

## Don't pre-convolve?
im2_preconv = im2

from numpy.polynomial.chebyshev import chebval

# Parameters from stack
sigGauss = [0.75, 1.5, 3.0]
degGauss = [4, 2, 2]
betaGauss = 1   # in the Becker et al. paper sigGauss is 1 but PSF is more like 2 pixels?
spatialKernelOrder = 0 #2  # polynomial for modifying the shape of the kernel across the image
spatialBackgroundOrder = 0 #1  # polynomial for adding background gradient to fit
# Parameters from and Becker et al. (2012)
#sigGauss = [0.75, 1.5, 3.0]
#degGauss = [6, 4, 2]

def chebBasis(x, ord):
    coef = np.zeros(ord+1)
    coef[-1] = 1
    ch = chebval(x, coef)
    return ch, coef

def chebGauss(x, m=0., s=1., ord=0, beta=1.):
    ga = gaussian(x, m, s/beta)
    #ga /= ga.sum()
    ch, coef = chebBasis(x, ord)
    #ch /= ch.sum()
    ch -= ch.min()
    print s, ord, coef #, sum(ga), sum(ch), sum(ga*ch)
    out = ga * ch
    #out /= out.sum()
    return out

x = np.arange(-6, 6, 0.1)
basis = [chebGauss(x, m=0, s=sig, ord=deg, beta=betaGauss)          for i,sig in enumerate(sigGauss) for deg in range(degGauss[i])] #, kernelOrd=ko) for ko in range(spatialKernelOrder+1)]
basis = np.vstack(basis).T  # put the bases into columns

# Single call to do it with all bases
# First use the original (non spatially modified) basis
basis2 = [np.convolve(im1, b, mode='same') - im1 for b in basis.T]
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

get_ipython().magic('timeit np.linalg.lstsq(basis2, im2_preconv)')
pars_old, resid, _, _ = np.linalg.lstsq(basis2, im2_preconv)
print pars_old

get_ipython().run_cell_magic('timeit', '', 'b = np.dot(basis2.T, im2_preconv) #(basis2.T * im2_preconv).sum(1)\nM = np.dot(basis2.T, basis2)\nnp.linalg.lstsq(M, b)')

b = np.dot(basis2.T, im2_preconv) #(basis2.T * im2_preconv).sum(1)
M = np.dot(basis2.T, basis2)
pars, resid, _, _ = np.linalg.lstsq(M, b)
print pars
print pars-pars_old

fit = (pars * basis2).sum(1)
print basis2.shape, fit.shape, pars.shape
#plt.plot(x, im2 - fit)  # science - convolved template (red)
#plt.plot(x, im2 - im1)  # science - original template (blue)
plt.rcParams['figure.figsize'] = (12.0, 4.0)
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(xim, im1)  # original template (red)
ax1.plot(xim, fit)  # convolved template (blue)  -- note looks purple because it's right on top of im2
ax1.plot(xim, im2_preconv, ls='-.', lw=3)  # preconvolved science image (dotted, purple)
ax1.plot(xim, im2_preconv - fit)  # diffim (grey)

ax2.plot(xim, im2_preconv - fit)
plt.show()

print np.sum((im2_preconv-fit)**2), np.sum((im2-im1)**2)

kfit = (pars * basis).sum(1)
print kfit.sum()
kfit /= kfit.sum()
plt.plot(x, kfit)  # this plots the matching kernel

conv_im1 = np.convolve(im1, kfit, mode='same')
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(xim, im1)  # original template (red)
ax1.plot(xim, conv_im1)  # convolved template (blue)
ax1.plot(xim, im2_preconv, ls='-.', lw=3)  # science image (dotted, purple)
ax1.plot(xim, im2_preconv - conv_im1)  # diffim (grey)

ax2.plot(xim, im2_preconv - conv_im1)  # diffim
plt.show()

## Compute the "L(ZOGY)" post-conv. kernel from kfit

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
df.plot.hist(alpha=0.5, bins=20)
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
df.plot.hist(alpha=0.5, bins=40)

print tmp1.std()*5., tmp2.std()*5.
print np.sum(np.abs(tmp1) > tmp1.std()*5.), np.sum(np.abs(tmp2) > tmp2.std()*5.)

import scipy.stats
tmp1a, low, upp = scipy.stats.sigmaclip(tmp1)
tmp2a, low, upp = scipy.stats.sigmaclip(tmp2)
print tmp1a.std()*5., tmp2a.std()*5.

det1 = xim[np.abs(tmp1) > tmp1a.std()*5.]
det2 = xim[np.abs(tmp2) > tmp2a.std()*5.]
print len(det1), det1.min(), det1.max()
print len(det2), det2.min(), det2.max()

plt.plot(np.linspace(-0.02, 0.06), 15*gaussian(np.linspace(-0.02, 0.06), s=tmp1a.std()), color='r')
plt.plot(np.linspace(-0.02, 0.06), 15*gaussian(np.linspace(-0.02, 0.06), s=tmp2a.std()), color='b')

plt.plot(xim, tmp1)  # red - corrected
plt.plot(det1, np.repeat(tmp1.max(), len(det1)), '|', color='r')
plt.plot([-2000, 2000], np.repeat(tmp1a.std()*5., 2), color='r')
plt.plot(xim, tmp2)  # blue - original
plt.plot(det2, np.repeat(tmp2.max(), len(det2)), '|', color='b')
plt.plot([-2000, 2000], np.repeat(tmp2a.std()*5., 2), color='b')
plt.xlim(-20, 20)




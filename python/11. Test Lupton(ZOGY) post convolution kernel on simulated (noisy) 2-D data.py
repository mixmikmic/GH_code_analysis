import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14.0, 4.0)

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

from scipy.stats import multivariate_normal

## see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

# def gaussian2d(grid, m=None, s=None):
#     if m is None:
#         m = [0., 0.]
#     if s is None:
#         s = [1., 1.]
#     cov = [[s[0], 0], [0, s[1]]]
#     var = multivariate_normal(mean=m, cov=cov)
#     return var.pdf(grid)

def singleGaussian2d(x, y, xc, yc, sigma_x=1., sigma_y=1., theta=0., offset=0.):
    cos_theta2, sin_theta2 = np.cos(theta)**2., np.sin(theta)**2.
    sigma_x2, sigma_y2 = sigma_x**2., sigma_y**2.
    a = cos_theta2/(2.*sigma_x2) + sin_theta2/(2.*sigma_y2)
    b = -(np.sin(2.*theta))/(4.*sigma_x2) + (np.sin(2.*theta))/(4.*sigma_y2)
    c = sin_theta2/(2.*sigma_x2) + cos_theta2/(2.*sigma_y2)
    xxc, yyc = x-xc, y-yc
    out = np.exp( - (a*(xxc**2.) + 2.*b*xxc*yyc + c*(yyc**2.)))
    if offset != 0.:
        out += offset
    return out

np.random.seed(66)

imsize = 1024  # 2048
xim = np.arange(-imsize/2+1, imsize/2, 1.0)  # assume image coords are centered on sources
yim = xim.copy()
y0, x0 = np.meshgrid(xim, yim)
#grid = np.dstack((x0, y0))

n_sources = 50
psf1 = 1.6 # sigma in pixels im1 will be template
psf2 = 2.2 # sigma in pixels im2 will be science image
source_x_t = np.random.uniform(xim.min()+20, xim.max()-20, size=n_sources)
source_y_t = np.random.uniform(yim.min()+20, yim.max()-20, size=n_sources)
source_x_s = source_x_t - 0.1   # add an offset?
source_y_s = source_y_t - 0.1   # add an offset?
source_flux_t = np.random.uniform(500, 30000, size=n_sources)
source_flux_s = source_flux_t.copy()  # fluxes of sources in science im
xy_changed = np.argmin(np.abs(source_x_t**2 + source_y_t**2))  # set the source that changes to be the one closest to the center of the image
x_changed = source_x_t[xy_changed]
y_changed = source_y_t[xy_changed]
source_flux_s[xy_changed] *= 1.5  # make the source closest to x=0 have a small change in flux
xcen = int(source_x_t[xy_changed])
ycen = int(source_y_t[xy_changed])
print xcen, ycen, source_flux_s[xy_changed]

xbas = np.arange(-15, 16, 1)
ybas = xbas.copy()
y0bas, x0bas = np.meshgrid(xbas, ybas)

im1 = np.zeros(x0.shape)
im2 = im1.copy()
source1 = singleGaussian2d(x0bas, y0bas, 0, source_y_t[0], sigma_x=psf1, sigma_y=psf1)
for i, cen in enumerate(source_x_t): # This could definitely be sped up.
    cenx = source_x_t[i] - np.floor(source_x_t[i])
    ceny = source_y_t[i] - np.floor(source_y_t[i])
    source = singleGaussian2d(x0bas, y0bas, cenx, ceny, sigma_x=psf1, sigma_y=psf1)
    source *= (source_flux_t[i] / source.sum())
    im1[(np.int(source_x_t[i])+imsize/2-16):(np.int(source_x_t[i])+imsize/2+15), 
        (np.int(source_y_t[i])+imsize/2-16):(np.int(source_y_t[i])+imsize/2+15)] += source

    cenx = source_x_s[i] - np.floor(source_x_s[i])
    ceny = source_y_s[i] - np.floor(source_y_s[i])
    source = singleGaussian2d(x0bas, y0bas, cenx, ceny, sigma_x=psf2, sigma_y=psf2)
    source *= (source_flux_s[i] / source.sum())
    im2[(np.int(source_x_s[i])+imsize/2-16):(np.int(source_x_s[i])+imsize/2+15), 
        (np.int(source_y_s[i])+imsize/2-16):(np.int(source_y_s[i])+imsize/2+15)] += source

sig1 = 0.2  # sigma of template
sig2 = 0.2  # sigma of science image
im1_noise = np.random.normal(scale=sig1, size=im1.shape)
im2_noise = np.random.normal(scale=sig2, size=im2.shape)
print source_flux_t.sum()
print im1.sum(), im2.sum(), im1[xcen+imsize/2, ycen+imsize/2]
im1 += im1_noise
im2 += im2_noise
print im1.sum(), im2.sum(), im1[xcen+imsize/2, ycen+imsize/2]

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(1, (15, 5))
igrid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.0, share_all=True, label_mode="L",
                    cbar_location="top", cbar_mode="single")
extent = (xcen-200+imsize/2, xcen+200+imsize/2, ycen-200+imsize/2, ycen+200+imsize/2)
gim = igrid[0].imshow(im1[extent[0]:extent[1],extent[2]:extent[3]], clim=(0,20), origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[1].imshow(im2[extent[0]:extent[1],extent[2]:extent[3]], clim=(0,20), origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[2].imshow((im2-im1)[extent[0]:extent[1],extent[2]:extent[3]], clim=(-20,20), origin='lower', interpolation='none', cmap='gray', extent=extent)
#igrid.cbar_axes[0].colorbar(gim)

from numpy.polynomial.chebyshev import chebval2d

# Parameters from stack
sigGauss = [0.75, 1.5, 3.0]
degGauss = [4, 2, 2]
betaGauss = 1   # in the Becker et al. paper sigGauss is 1 but PSF is more like 2 pixels?
# Parameters from and Becker et al. (2012)
#sigGauss = [0.75, 1.5, 3.0]
#degGauss = [6, 4, 2]

from scipy.stats import multivariate_normal

## see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

def gaussian2d(grid, m=None, s=None):
    if m is None:
        m = [0., 0.]
    if s is None:
        s = [1., 1.]
    cov = [[s[0], 0], [0, s[1]]]
    var = multivariate_normal(mean=m, cov=cov)
    return var.pdf(grid)

def chebGauss2d(x, y, grid, m=None, s=None, ord=[0,0], beta=1., verbose=False):
    if m is None:
        m = [0., 0.]
    if s is None:
        s = [1., 1.]
    cov = [[s[0], 0], [0, s[1]]]
    coefLen = np.max(ord)+1
    coef0 = np.zeros(coefLen)
    coef0[ord[0]] = 1
    coef1 = np.zeros(coefLen)
    coef1[ord[1]] = 1
    if verbose:
        print s, ord, coef0, coef1
    ga = gaussian2d(grid, m, np.array(s)/beta)
    #ga = singleGaussian2d(x, y, xc=m[0], yc=m[1], sigma_x=s[0]/beta, sigma_y=s[1]/beta)
    #ga /= ga.sum()
    ch = chebval2d(x, y, c=np.outer(coef0, coef1))
    return ch * ga

# Set the coordinates for the bases
xbas = np.arange(-15, 16, 1)
ybas = xbas.copy()
y0bas, x0bas = np.meshgrid(xbas, ybas)
gridbas = np.dstack((y0bas, x0bas))

basis = [chebGauss2d(x0bas, y0bas, gridbas, m=[0,0], s=[sig0,sig1], ord=[deg0,deg1], beta=betaGauss, verbose=True) for i0,sig0 in enumerate(sigGauss) for i1,sig1 in enumerate(sigGauss) for deg0 in range(degGauss[i0]+1) for deg1 in range(degGauss[i1]+1)]
print len(basis), basis[0].shape, x0bas.shape, basis[0].reshape(x0bas.shape).shape

print len(basis), basis[0].shape, basis[0].reshape(x0bas.shape).shape, 31*31
basis2 = np.dstack(basis).T  # put the bases into an array
print basis2.shape, basis2[0].shape, basis2[0].reshape(x0bas.shape).shape
print basis[0].min(), basis[0].max(), basis2.min(), basis2.max()

limits = (-0.2, 0.2) #(basis2.min(), basis2.max())
fig = plt.figure(1, (16., 16.))
igrid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(11, 11),  # creates 2x2 grid of axes
                    axes_pad=0.0, share_all=True, label_mode="L", cbar_location="top", cbar_mode="single")
extent = (x0bas.min()+10, x0bas.max()-10, y0bas.min()+10, y0bas.max()-10)
for i in range(121):
    gim = igrid[i].imshow(basis[i].reshape(x0bas.shape), origin='lower', interpolation='none', cmap='gray', 
                          extent=extent, clim=limits)
igrid.cbar_axes[0].colorbar(gim)

if False:
    im2_psf = gaussian(x, s=psf2)
    im2_psf /= im2_psf.sum()
    im2_preconv = np.convolve(im2, im2_psf, mode='same')
    print im2.sum(), im2_preconv.sum()
    plt.plot(xim, im1); plt.plot(xim, im2); plt.plot(xim, im2_preconv); plt.plot(xim, im2-im1)

## Don't pre-convolve?
im2_preconv = im2

import scipy.ndimage.filters
import scipy.signal

print im2.shape, basis[3].shape
tmp1 = scipy.ndimage.filters.convolve(im1, basis[3], mode='constant')
tmp2 = scipy.signal.fftconvolve(im1, basis[3], mode='same')
print tmp1.shape, tmp1.min(), tmp1.max()
print tmp2.shape, tmp2.min(), tmp2.max()
print (tmp1-tmp2).std(), (tmp1-tmp2).mean()

# Single call to do it with all bases
# First use the original (non spatially modified) basis
def tmpfun(im, b, i):
    #print i
    #out = scipy.ndimage.filters.convolve(im, b, mode='constant')
    out = scipy.signal.fftconvolve(im, b, mode='same')
    return out

basis2 = [tmpfun(im1, b, i) for i,b in enumerate(basis)]
print len(basis2), basis2[0].shape

basis2a = np.vstack([b.flatten() for b in basis2]).T
print basis2a.shape, im2.flatten().shape

#%timeit np.linalg.lstsq(basis2, im2_preconv)
pars_old, resid, _, _ = np.linalg.lstsq(basis2a, im2_preconv.flatten())
print pars_old[:8]

fit_old = (pars_old * basis2a).sum(1).reshape(im2_preconv.shape)
print resid, np.sum((im2_preconv - fit_old.reshape(im2_preconv.shape))**2)
print scipy.stats.describe((im2_preconv - im1)**2, axis=None)
print scipy.stats.describe((im2_preconv - fit_old)**2, axis=None)
print basis2a.shape, fit_old.shape

kbasis = np.vstack([b.flatten() for b in basis]).T
print pars_old.shape, kbasis.shape
kfit = (pars_old * kbasis).sum(1).reshape(basis[0].shape)
print kfit.sum()
kfit /= kfit.sum()
#extent = (x0.min()+10, x0.max()-10, y0.min()+10, y0.max()-10)
plt.imshow(kfit, interpolation='none', cmap='gray')  # this plots the matching kernel

b = (basis2a.T * im2_preconv.flatten()).sum(1)
print b.shape

M = np.dot(basis2a.T, basis2a)
print M.shape

pars2, resid, _, _ = np.linalg.lstsq(M, b)
print pars2[:8]

fit2 = (pars2 * basis2a).sum(1).reshape(im2_preconv.shape)
print basis2a.shape, fit2.shape, fit2.min(), fit2.max()
print scipy.stats.describe((im2_preconv - im1)**2, axis=None)
print scipy.stats.describe((im2_preconv - fit2)**2, axis=None)

fig = plt.figure(1, (16, 4))
extent = (xcen-20+imsize/2, xcen+20+imsize/2, ycen-20+imsize/2, ycen+20+imsize/2)
igrid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.0)
extent = (xcen-200+imsize/2, xcen+200+imsize/2, ycen-200+imsize/2, ycen+200+imsize/2)
gim = igrid[0].imshow(im1[extent[0]:extent[1],extent[2]:extent[3]], clim=(0,20), origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[1].imshow(fit2[extent[0]:extent[1],extent[2]:extent[3]], clim=(-0.2,20), origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[2].imshow((im2_preconv-im1)[extent[0]:extent[1],extent[2]:extent[3]], clim=(-2,2), origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[3].imshow((im2_preconv-fit2)[extent[0]:extent[1],extent[2]:extent[3]], clim=(-2,2), origin='lower', interpolation='none', cmap='gray', extent=extent)

kbasis = np.vstack([b.flatten() for b in basis]).T
print pars2.shape, kbasis.shape
kfit2 = (pars2 * kbasis).sum(1).reshape(basis[0].shape)
print kfit2.sum()
kfit2 /= kfit2.sum()
#extent = (x0.min()+10, x0.max()-10, y0.min()+10, y0.max()-10)
plt.imshow(kfit2, interpolation='none', cmap='gray')  # this plots the matching kernel

conv_im1 = scipy.ndimage.filters.convolve(im1, kfit2, mode='constant')
print conv_im1.shape, conv_im1.min(), conv_im1.max()
print scipy.stats.describe((im2_preconv - im1)**2, axis=None)
print scipy.stats.describe((im2_preconv - conv_im1)**2, axis=None)

fig = plt.figure(1, (16, 4))
extent = (xcen-20+imsize/2, xcen+20+imsize/2, ycen-20+imsize/2, ycen+20+imsize/2)
igrid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.0)
extent = (xcen-200+imsize/2, xcen+200+imsize/2, ycen-200+imsize/2, ycen+200+imsize/2)
gim = igrid[0].imshow(im1[extent[0]:extent[1],extent[2]:extent[3]], clim=(0,20), origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[1].imshow(conv_im1[extent[0]:extent[1],extent[2]:extent[3]], clim=(0,20), origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[2].imshow((im2_preconv-im1)[extent[0]:extent[1],extent[2]:extent[3]], clim=(-10,10), origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[3].imshow((im2_preconv-conv_im1)[extent[0]:extent[1],extent[2]:extent[3]], clim=(-1,2), origin='lower', interpolation='none', cmap='gray', extent=extent)

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




import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

x = np.arange(-16, 16, 1)
y = x.copy()
y0, x0 = np.meshgrid(x, y)
grid = np.dstack((y0, x0))

import scipy.stats

## see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

def gaussian2d(grid, m=None, s=None):
    if m is None:
        m = [0., 0.]
    if s is None:
        s = [1., 1.]
    cov = [[s[0], 0], [0, s[1]]]
    var = scipy.stats.multivariate_normal(mean=m, cov=cov)
    return var.pdf(grid)

#g1 = gaussian2d(grid) + gaussian2d(grid, [5,5], [3,1])
#plt.imshow(g1)

np.random.seed(66)

xim = np.arange(-256, 256, 1)
yim = xim.copy()
y0im, x0im = np.meshgrid(xim, yim)
imgrid = np.dstack((y0im, x0im))
n_sources = 500
fluxes = np.random.uniform(50, 30000, n_sources)
xposns = np.random.uniform(xim.min()+5, xim.max()-5, n_sources)
yposns = np.random.uniform(yim.min()+5, yim.max()-5, n_sources)

# Make the source closest to the center of the image the one that increases in flux
ind = np.argmin(xposns**2. + yposns**2.)
print ind, xposns[ind], yposns[ind]

sig1 = 0.2  # sigma of template
sig2 = 0.2  # sigma of science image
im1 = np.random.normal(scale=sig1, size=x0im.shape)
im2 = np.random.normal(scale=sig2, size=x0im.shape)

psf1 = 1.6 # sigma in pixels im1 will be template
psf2 = 2.2 # sigma in pixels im2 will be science image. make the psf in this image slighly offset and elongated
offset = 0.0  # 0.2  # astrometric offset in pixels between two images
for i in range(n_sources):
    im1 += fluxes[i] * gaussian2d(imgrid, m=[xposns[i],yposns[i]], s=[psf1, psf1])
    im2 += fluxes[i] * gaussian2d(imgrid, m=[xposns[i]+offset,yposns[i]+offset], s=[psf2, psf2*1.5])
    if i == ind:
        im2 += (fluxes[i]/50.) * gaussian2d(imgrid, m=[xposns[i]+offset,yposns[i]+offset], s=[psf2, psf2*1.5])

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(1, (9, 3))
igrid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.0, share_all=True, label_mode="L",
                    cbar_location="top", cbar_mode="single")
extent = (x0im.min()+150, x0im.max()-150, y0im.min()+150, y0im.max()-150)
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
gim = igrid[0].imshow(im1[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[1].imshow(im2[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[2].imshow((im2-im1)[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent, clim=(-10,10))
igrid.cbar_axes[0].colorbar(gim)
print scipy.stats.describe(im2 - im1, axis=None)

from scipy.fftpack import fft2, ifft2, fftfreq, fftshift

# First compute the science image's (im2's) psf -- easy since we parameterized it above when we made the image
im1_psf = gaussian2d(grid, m=[0, 0], s=[psf1, psf1])
im2_psf = gaussian2d(grid, m=[0, 0], s=[psf2, psf2*1.5])

im1_psf_ft = fft2(im1_psf)
im2_psf_ft = fft2(im2_psf)

im1_ft = fft2(im1)
im2_ft = fft2(im2)

# Just a sanity check:
fig = plt.figure(1, (8, 4))
igrid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.0, share_all=True, label_mode="L",
                    cbar_location="top", cbar_mode="single")
extent = (x0im.min()+150, x0im.max()-150, y0im.min()+150, y0im.max()-150)
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
gim = igrid[0].imshow(im1[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[1].imshow(ifft2(im1_ft).real[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid.cbar_axes[0].colorbar(gim)
print scipy.stats.describe(im1, axis=None)
print scipy.stats.describe(ifft2(im1_ft).real, axis=None)

F_r = F_n = 1.
R_hat = fft2(im1)
N_hat = fft2(im2)
P_r = gaussian2d(imgrid, m=[0, 0], s=[psf1, psf1])
P_n = gaussian2d(imgrid, m=[0, 0], s=[psf2, psf2*1.5])
P_r_hat = fft2(P_r)
P_n_hat = fft2(P_n)
d_hat_numerator = (F_r * P_r_hat * N_hat - F_n * P_n_hat * R_hat)
d_hat_denom = np.sqrt((sig1**2 * F_r**2 * np.abs(P_r_hat)**2) + (sig2**2 * F_n**2 * np.abs(P_n_hat)**2))
d_hat = d_hat_numerator / d_hat_denom

d = ifft2(d_hat)
D = np.fft.ifftshift(d.real)
plt.imshow(D[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent, clim=(-4, 4))
print scipy.stats.describe(D, axis=None)

F_D_numerator = F_r * F_n
F_D_denom = np.sqrt(sig1**2 * F_r**2 + sig2**2 * F_n**2)
F_D = F_D_numerator / F_D_denom

P_d_hat_numerator = (F_r * F_n * P_r_hat * P_n_hat)
P_d_hat = P_d_hat_numerator / (F_D * d_hat_denom)

P_d = ifft2(P_d_hat)
P_D = np.fft.ifftshift(P_d.real)
extent = (x0im.min()+240, x0im.max()-240, y0im.min()+240, y0im.max()-240)
x1p, x2p, y1p, y2p = 240, 511-240, 240, 511-240   # limits for display
plt.imshow(P_D[x1p:x2p,y1p:y2p], origin='lower', interpolation='none', cmap='gray', extent=extent) #, clim=(0, 0.67))
print scipy.stats.describe(P_D, axis=None)

import cPickle
import gzip
cPickle.dump((im1, im2, im1_psf, im2_psf, D, P_D), gzip.GzipFile("12_results.p.gz", "wb"))

from scipy.fftpack import fft2, ifft2, fftfreq, fftshift, ifftshift

# Note if we use a smaller PSF (e.g. 32x32) then we get artifacts in the convolution
psfSize = 25
x = np.arange(-psfSize+1, psfSize, 1)
y = x.copy()
y0, x0 = np.meshgrid(x, y)
grid = np.dstack((y0, x0))

psf1 = 1.6 #* 5.
psf2 = 2.2 #* 5.
sig1 = sig2 = 0.2
F_r = F_n = 1.0
P_r = gaussian2d(grid, m=[0, 0], s=[psf1, psf1])
P_n = gaussian2d(grid, m=[0, 0], s=[psf2, psf2*1.5])
print sig1, sig2, psf1, psf2
print sig1, sig2, P_r.sum(), P_n.sum()
P_r_hat = fft2(P_r)
P_n_hat = fft2(P_n)
denom = np.sqrt((sig1**2 * F_r**2 * np.abs(P_r_hat)**2) + (sig2**2 * F_n**2 * np.abs(P_n_hat)**2))
K_r_hat = P_r_hat / denom
K_n_hat = P_n_hat / denom
print np.abs(K_r_hat).max(), np.abs(K_r_hat).min(), np.abs(K_r_hat).sum()
#K_r = np.fft.ifftshift(np.abs(ifft2(K_r_hat)).real)
#K_n = np.fft.ifftshift(np.abs(ifft2(K_n_hat)).real)
K_r = ifft2(K_r_hat).real
K_n = ifft2(K_n_hat).real

pck = K_r.real #ifftshift(K_n).real
print K_r.shape, pck.real.shape
print pck.max(), pck.min(), pck.sum()
print np.unravel_index(np.argmax(pck), pck.shape)
print np.unravel_index(np.argmax(P_r), P_r.shape)

import diffimTests as dit
reload(dit)

dit.plotImageGrid((K_r.real, K_n.real), clim=(-0.04,0.05))

import scipy.ndimage.filters

tmp = K_n.real #[1:,1:]
im1c = scipy.ndimage.filters.convolve(im1, tmp, mode='constant')
tmp = K_r.real #[1:,1:]
im2c = scipy.ndimage.filters.convolve(im2, tmp, mode='constant')
diffim = im2c - im1c
print scipy.stats.describe(diffim, axis=None)

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(1, (8, 4))
igrid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.0, share_all=True, label_mode="L",
                    cbar_location="top", cbar_mode="single")
extent = (x0im.min()+150, x0im.max()-150, y0im.min()+150, y0im.max()-150)
#x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
x1d, x2d, y1d, y2d = 240, 512-240, 240, 512-240   # limits for display
#plt.imshow(diffim[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent, clim=(-2.5,2.5)) # corrected diffim
gim = igrid[0].imshow(D[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent, clim=(-1,1))
igrid[1].imshow(diffim[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent, clim=(-1,1))
igrid.cbar_axes[0].colorbar(gim)

d1 = D
d2 = diffim
# The ZOGY diffim has artefacts on the edges. Let's set them to zero so they dont mess up the stats.
# The ZOGY(real) diffim has zero's at 32 pixels surrounding the image.
# Actually, let's just set the edge pixels of both diffims to zero.
d2[d1 == 0.] = 0.
d1[0,:] = d1[:,0] = d1[-1,:] = d1[:,-1] = 0.
d2[0,:] = d2[:,0] = d2[-1,:] = d2[:,-1] = 0.

import scipy.stats
_, low, upp = scipy.stats.sigmaclip([d1, d2])
print low, upp
low *= 1.1
upp *= 1.1
d2a = d2[(d1>low) & (d1<upp) & (d2>low) & (d2<upp) & (d2!=0.) & (d1!=0.)]
d1a = d1[(d1>low) & (d1<upp) & (d2>low) & (d2<upp) & (d1!=0.)]
print len(d1a.flatten()), len(d2a.flatten())
df = pd.DataFrame({'L(ZOGY)': d1a.flatten()/d1a.std(), 'ZOGY': d2a.flatten()/d2a.std()})
df.plot.hist(alpha=0.5, bins=200)

print d1a.std(), d2a.std()
print d1.max(), d2.max()
print d1.max()/d1a.std(), d2.max()/d2a.std()
print np.sum(d1>d1a.std()*5.), np.sum(d2>d2a.std()*5.)

plt.imshow(d1 - d2, origin='lower', interpolation='none', cmap='gray', clim=(-0.1,0.1))

df = pd.DataFrame({'diff': (d1 - d2).flatten()})
df.plot.hist(alpha=0.5, bins=10000)
print df.min()[0], df.max()[0], df.std()[0]
plt.xlim(-0.002, 0.002)

reload(dit)

# Let's try w same parameters as ZOGY paper.
sky = 300.

im1, im2, im1_psf, im2_psf, im1_var, im2_var, changedCentroid =     dit.makeFakeImages(xim=None, yim=None, sky=sky, psf1=None, psf2=None, offset=[0., 0.],
                    psf_yvary_factor=0., varSourceChange=1500., theta1=0., theta2=-45., im2background=0.,
                    n_sources=50, sourceFluxRange=(500,30000), seed=66, psfSize=None)
    
_, _, P_r, P_n, _, _, changedCentroid =     dit.makeFakeImages(xim=None, yim=None, sky=sky, psf1=None, psf2=None, offset=[0., 0.],
                    psf_yvary_factor=0., varSourceChange=1500., theta1=0., theta2=-45., im2background=0.,
                    n_sources=50, sourceFluxRange=(500,30000), seed=66, psfSize=25)

print dit.computeClippedImageStats(im1)
print dit.computeClippedImageStats(im2)
print dit.computeClippedImageStats(im1_var)
print dit.computeClippedImageStats(im2_var)

reload(dit)
D = dit.performZOGY(im1, im2, im1_psf, im2_psf)
D_AL, _ = dit.performAlardLupton(im1, im2, spatialKernelOrder=0, spatialBackgroundOrder=1)
D_AL /= np.sqrt(sky * 2.)
D_new = dit.performZOGYImageSpace(im1, im2, P_r, P_n)
print dit.computeClippedImageStats(D)
print dit.computeClippedImageStats(D_new)
print dit.computeClippedImageStats(D_AL)

fig = plt.figure(1, (12, 12))
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
extent = (x0im.min()+150, x0im.max()-150, y0im.min()+150, y0im.max()-150)
dit.plotImageGrid((im1[x1d:x2d,y1d:y2d], D[x1d:x2d,y1d:y2d], D_new[x1d:x2d,y1d:y2d], D_AL[x1d:x2d,y1d:y2d]), 
                  clim=(-3,3), titles=['Template', 'ZOGY', 'ZOGY(image)', 'A&L'])

D_new[0,:] = D_new[:,0] = D_new[-1,:] = D_new[:,-1] = 0.
D[D_new == 0] = 0.
D_AL[D_new == 0] = 0.
fig = plt.figure(1, (8, 8))
dit.plotImageGrid(((D - D_new)[x1d:x2d,y1d:y2d], (D_new - D_AL)[x1d:x2d,y1d:y2d]), clim=(-0.1, 0.1))

P_D, F_D = dit.computeZOGYDiffimPsf(im1, im2, P_r, P_n)
print F_D
dit.plotImageGrid((P_D, P_r, P_n), clim=(-0.01,0.01))

reload(dit);

S_corr, S, D, P_D, F_D = dit.performZOGY_Scorr(im1, im2, im1_var, im2_var, P_r, P_n)
print dit.computeClippedImageStats(S_corr)
print dit.computeClippedImageStats(S)
#dit.plotImageGrid((S_corr, S, var1c, var2c), clim=(-2,2))
print changedCentroid, S_corr[np.rint(changedCentroid[1]).astype(int), np.rint(changedCentroid[0]).astype(int)]
print (S_corr > 5.).sum() + (S_corr < -5.).sum()
fig = plt.figure(1, (12, 12))
dit.plotImageGrid((S_corr, ((S_corr > 5.)*5.0 + (S_corr < -5.)*-5.0)), clim=(-5.,5.))

S_corr, S, D, P_D, F_D = dit.performZOGY_Scorr(im1, im2, im1_var, im2_var, P_r, P_n, D=D_AL)
print dit.computeClippedImageStats(S_corr)
print dit.computeClippedImageStats(S)
#dit.plotImageGrid((S_corr, S, var1c, var2c), clim=(-2,2))
print changedCentroid, S_corr[np.rint(changedCentroid[1]).astype(int), np.rint(changedCentroid[0]).astype(int)]
print (S_corr > 5.).sum() + (S_corr < -5.).sum()
fig = plt.figure(1, (12, 12))
dit.plotImageGrid((S_corr, ((S_corr > 5.)*5.0 + (S_corr < -5.)*-5.0)), clim=(-5.,5.))




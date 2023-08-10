import cPickle
import gzip
# First from notebook 11 L(ZOGY) data: (im1, im2, im1_psf, im2_psf, conv_im1, pci, pcf)
LZOGY = cPickle.load(gzip.GzipFile("11_results.p.gz", "rb"))
# Next from notebook 12 ZOGY data: (im1, im2, im1_psf, im2_psf, D, P_D)
ZOGY = cPickle.load(gzip.GzipFile("12_results.p.gz", "rb"))

import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

x = np.arange(-15, 16, 1)
y = x.copy()
y0, x0 = np.meshgrid(x, y)
grid = np.dstack((y0, x0))

xim = np.arange(-255, 256, 1)
yim = xim.copy()
y0im, x0im = np.meshgrid(xim, yim)
imgrid = np.dstack((y0im, x0im))

# Make sure they're the same!!!
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(1, (9, 3))
igrid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.0, share_all=True, label_mode="L",
                    cbar_location="top", cbar_mode="single")
extent = (-255+150, 256-150, -255+150, 256-150)
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
gim = igrid[0].imshow(LZOGY[0][x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[1].imshow(ZOGY[0][x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent)
igrid[2].imshow((LZOGY[0]-ZOGY[0])[x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent, clim=(-1,1))
igrid.cbar_axes[0].colorbar(gim)

# Compare the optimal diffim's:
fig = plt.figure(1, (8, 4))
igrid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.0, share_all=True, label_mode="L",
                    cbar_location="top", cbar_mode="single")
extent = (-255+150, 256-150, -255+150, 256-150)
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
gim = igrid[0].imshow(LZOGY[5][x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent, clim=(-0.28,0.28))
igrid[1].imshow(ZOGY[4][x1d:x2d,y1d:y2d], origin='lower', interpolation='none', cmap='gray', extent=extent, clim=(-1,1))
igrid.cbar_axes[0].colorbar(gim)

d1 = LZOGY[5]
d2 = ZOGY[4]
# The ZOGY diffim has artefacts on the edges. Let's set them to zero so they dont mess up the stats.
# Actually, let's just set the edge pixels of both diffims to zero.
d1[0,:] = d1[:,0] = d1[-1,:] = d1[:,-1] = 0.
d2[0,:] = d2[:,0] = d2[-1,:] = d2[:,-1] = 0.

import scipy.stats
_, low, upp = scipy.stats.sigmaclip([d1, d2])
print low, upp
low *= 1.1
upp *= 1.1
d1a = d1[(d1>low) & (d1<upp) & (d2>low) & (d2<upp)]
d2a = d2[(d1>low) & (d1<upp) & (d2>low) & (d2<upp)]
df = pd.DataFrame({'L(ZOGY)': d1a.flatten()/d1a.std(), 'ZOGY': d2a.flatten()/d2a.std()})
df.plot.hist(alpha=0.5, bins=200)

print d1a.std(), d2a.std()
print d1.max(), d2.max()
print d1.max()/d1a.std(), d2.max()/d2a.std()
print np.sum(d1>d1a.std()*5.), np.sum(d2>d2a.std()*5.)

plt.imshow(d1/d1a.std() - d2/d2a.std(), origin='lower', interpolation='none', cmap='gray', clim=(-0.1,0.1))

df = pd.DataFrame({'diff': (d1/d1a.std() - d2/d2a.std()).flatten()})
df.plot.hist(alpha=0.5, bins=2000)
print df.min()[0], df.max()[0]
plt.xlim(-0.1, 0.1)

print (d1/d1a.std() - d2/d2a.std()).std()

diffim = LZOGY[1] - LZOGY[4]
plt.imshow(diffim/diffim.std(), origin='lower', interpolation='none', cmap='gray', clim=(-0.1,0.1))

def get_covariance(diffim):
    #diffim = diffim/diffim.std()
    shifted_imgs = [
        diffim,
        np.roll(diffim, 1, 0), np.roll(diffim, -1, 0), np.roll(diffim, 1, 1), np.roll(diffim, -1, 1),
        np.roll(np.roll(diffim, 1, 0), 1, 1), np.roll(np.roll(diffim, 1, 0), -1, 1),
        np.roll(np.roll(diffim, -1, 0), 1, 1), np.roll(np.roll(diffim, -1, 0), -1, 1),
        np.roll(diffim, 2, 0), np.roll(diffim, -2, 0), np.roll(diffim, 2, 1), np.roll(diffim, -2, 1),
        np.roll(diffim, 3, 0), np.roll(diffim, -3, 0), np.roll(diffim, 3, 1), np.roll(diffim, -3, 1),
        np.roll(diffim, 4, 0), np.roll(diffim, -4, 0), np.roll(diffim, 4, 1), np.roll(diffim, -4, 1),
        np.roll(diffim, 5, 0), np.roll(diffim, -5, 0), np.roll(diffim, 5, 1), np.roll(diffim, -5, 1),
        #np.roll(np.roll(diffim, 2, 0), 1, 1), np.roll(np.roll(diffim, 2, 0), -1, 1),
        #np.roll(np.roll(diffim, -2, 0), 1, 1), np.roll(np.roll(diffim, -2, 0), -1, 1),
        #np.roll(np.roll(diffim, 1, 0), 2, 1), np.roll(np.roll(diffim, 1, 0), -2, 1),
        #np.roll(np.roll(diffim, -1, 0), 2, 1), np.roll(np.roll(diffim, -1, 0), -2, 1),
        #np.roll(np.roll(diffim, 2, 0), 2, 1), np.roll(np.roll(diffim, 2, 0), -2, 1),
        #np.roll(np.roll(diffim, -2, 0), 2, 1), np.roll(np.roll(diffim, -2, 0), -2, 1),
    ]
    shifted_imgs = np.vstack([i.flatten() for i in shifted_imgs])
    out = np.corrcoef(shifted_imgs)
    out = np.cov(shifted_imgs, bias=1)
    tmp2 = out.copy()
    np.fill_diagonal(tmp2, np.NaN)
    print np.nansum(tmp2)/np.sum(np.diag(out))  # print sum of off-diag / sum of diag
    return out#[inds, :][:, inds]

cov = get_covariance(diffim/diffim.std())
plt.imshow(cov, interpolation='none', clim=(0, 0.15))
plt.colorbar()

cov = get_covariance(d1/d1a.std())
plt.imshow(cov, interpolation='none', clim=(0, 0.15))
plt.colorbar()

cov = get_covariance(d2/d2a.std())
plt.imshow(cov, interpolation='none', clim=(0, 0.15))
plt.colorbar()

np.random.seed(66)
m1 = np.random.normal(scale=0.2, size=diffim.shape)
m2 = np.random.normal(scale=0.2, size=diffim.shape)
mx = m1 - m2
cov = get_covariance(mx/mx.std())
plt.imshow(cov, interpolation='none', clim=(0, 0.15))
plt.colorbar()




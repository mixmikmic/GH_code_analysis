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

import diffimTests as dit
reload(dit)

# Let's try w same parameters as ZOGY paper.
sky = 300.

# Generate images and PSF's with the same dimension as the image (used for A&L)
im1, im2, im1_psf, im2_psf, im1_var, im2_var, changedCentroid =     dit.makeFakeImages(imSize=(512, 512), sky=sky, psf1=None, psf2=None, offset=[0., 0.],
                    psf_yvary_factor=0., varSourceChange=1500., theta1=0., theta2=-45., im2background=0.,
                    n_sources=50, sourceFluxRange=(500,30000), seed=66, psfSize=None)

# This is a hack just to generate a pair of PSFs with size 50x50
_, _, P_r, P_n, _, _, changedCentroid =     dit.makeFakeImages(imSize=None, sky=sky, psf1=None, psf2=None, offset=[0., 0.],
                    psf_yvary_factor=0., varSourceChange=1500., theta1=0., theta2=-45., im2background=0.,
                    n_sources=5, sourceFluxRange=(500,30000), seed=66, psfSize=25)

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

xim = np.arange(-256, 256, 1)
yim = xim.copy()
y0im, x0im = np.meshgrid(xim, yim)
imgrid = np.dstack((y0im, x0im))

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

reload(dit)
D = dit.performZOGY(im2, im1, im2_psf, im1_psf)
D_AL, _ = dit.performAlardLupton(im2, im1, spatialKernelOrder=0, spatialBackgroundOrder=1)
D_AL /= np.sqrt(sky * 2.)
D_new = dit.performZOGYImageSpace(im2, im1, P_n, P_r)
print dit.computeClippedImageStats(D)
print dit.computeClippedImageStats(D_new)
print dit.computeClippedImageStats(D_AL)

xim = np.arange(-256, 256, 1)
yim = xim.copy()
y0im, x0im = np.meshgrid(xim, yim)
imgrid = np.dstack((y0im, x0im))

fig = plt.figure(1, (12, 12))
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
extent = (x0im.min()+150, x0im.max()-150, y0im.min()+150, y0im.max()-150)
dit.plotImageGrid((im1[x1d:x2d,y1d:y2d], -D[x1d:x2d,y1d:y2d], -D_new[x1d:x2d,y1d:y2d], -D_AL[x1d:x2d,y1d:y2d]), 
                  clim=(-3,3), titles=['Template', 'ZOGY', 'ZOGY(image)', 'A&L'])

reload(dit);

S_corr, S, D, P_D, F_D = dit.performZOGY_Scorr(im2, im1, im2_var, im1_var, P_n, P_r)
print dit.computeClippedImageStats(S_corr)
print dit.computeClippedImageStats(S)
#dit.plotImageGrid((S_corr, S, var1c, var2c), clim=(-2,2))
print changedCentroid, S_corr[np.rint(changedCentroid[1]).astype(int), np.rint(changedCentroid[0]).astype(int)]
print (S_corr > 5.).sum() + (S_corr < -5.).sum()
fig = plt.figure(1, (12, 12))
dit.plotImageGrid((-S_corr, ((S_corr > 5.)*-5.0 + (S_corr < -5.)*5.0)), clim=(-5.,5.))

reload(dit)

# Note kernelSize needs to equal psfSize in makeFakeImages call above

D_AL_pc, kappa = dit.performAlardLupton(im2, im1, spatialKernelOrder=0, spatialBackgroundOrder=1, 
                                        preConvKernel=P_n, doALZCcorrection=False)
D_AL_pc_ALZC, kappa2 = dit.performAlardLupton(im2, im1, spatialKernelOrder=0, spatialBackgroundOrder=1, 
                                        preConvKernel=P_n, doALZCcorrection=True)
print dit.computeClippedImageStats(D_AL_pc)
print dit.computeClippedImageStats(D_AL_pc_ALZC)

dit.plotImageGrid((kappa, P_n, kappa2), clim=(-0.01,0.01))

S_corr, S, D, P_D, F_D = dit.performZOGY_Scorr(im2, im1, im2_var, im1_var, P_n, P_r)
print dit.computeClippedImageStats(S)
print dit.computeClippedImageStats(S_corr)

xim = np.arange(-256, 256, 1)
yim = xim.copy()
y0im, x0im = np.meshgrid(xim, yim)
imgrid = np.dstack((y0im, x0im))

fig = plt.figure(1, (12, 12))
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
extent = (x0im.min()+150, x0im.max()-150, y0im.min()+150, y0im.max()-150)
dit.plotImageGrid((im1[x1d:x2d,y1d:y2d], -S_corr[x1d:x2d,y1d:y2d]*3., -D_AL_pc[x1d:x2d,y1d:y2d], 
                   -D_AL_pc_ALZC[x1d:x2d,y1d:y2d]), 
                  clim=(-30,30), titles=['Template', 'ZOGY', 'A&L (pre-filter, no decorr.)', 
                                         'A&L (pre-filter, with decorr.)'])

dstats = dit.computeClippedImageStats(D_AL_pc_ALZC)
print dstats
#D_AL_pc_ALZC /= 3.0
tmp = (D_AL_pc_ALZC - dstats[0]) / dstats[1]
print dit.computeClippedImageStats(tmp/S_corr)
print (tmp > 5.).sum() + (tmp < -5.).sum()
fig = plt.figure(1, (12, 12))
dit.plotImageGrid((-tmp, -(tmp-S_corr), ((tmp > 5.)*-5. + (tmp < -5.)*5.)), clim=(-5., 5.))




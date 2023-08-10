import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

varSourceFlux = 620. * np.sqrt(2.)
obj1 = dit.DiffimTest(varFlux2=np.repeat(varSourceFlux, 50), n_sources=500, sourceFluxRange=(200, 20000),
                     templateNoNoise=False, skyLimited=False)
obj2 = obj1.clone()

res1 = obj1.runTest(zogyImageSpace=False, returnSources=True)
df1, _ = obj1.doPlotWithDetectionsHighlighted(res1, transientsOnly=True, addPresub=True, 
                                              xaxisIsScienceForcedPhot=True,
                                              skyLimited=False, alpha=0.3)
del res1['sources']
print res1
plt.xlim(600, 1200)
plt.ylim(2, 12);

res2 = obj2.runTest(zogyImageSpace=True, returnSources=True)
df2, _ = obj2.doPlotWithDetectionsHighlighted(res2, transientsOnly=True, addPresub=True, 
                                              xaxisIsScienceForcedPhot=True,
                                              skyLimited=False, alpha=0.3)
del res2['sources']
print res2
plt.xlim(600, 1200)
plt.ylim(2, 12);

plt.scatter(df1[df1.ZOGY_detected == True].ZOGY_SNR.values,
            df2[df1.ZOGY_detected == True].ZOGY_SNR.values, c='r', alpha=0.2, label='Detected, fspace')
#ax = df[df.ZOGY_detected == True].plot.scatter('ZOGY_SNR', 'ALstack_decorr_SNR', c='r', alpha=0.2)
plt.scatter(df1[df1.ZOGY_detected == False].ZOGY_SNR.values,
            df2[df1.ZOGY_detected == False].ZOGY_SNR.values, c='k', s=10, alpha=0.7, label='Detected, none')
#df[df.ZOGY_detected == False].plot.scatter('ZOGY_SNR', 'ALstack_decorr_SNR', c='k', s=10, alpha=0.7, ax=ax)
plt.scatter(df1[df2.ZOGY_detected == True].ZOGY_SNR.values,
            df2[df2.ZOGY_detected == True].ZOGY_SNR.values, c='b', s=50, alpha=0.2, label='Detected, imspace')
plt.legend(loc='upper left')
#df[df.ALstack_decorr_detected == True].plot.scatter('ZOGY_SNR', 'ALstack_decorr_SNR', c='b', s=50, alpha=0.2, ax=ax)
#plt.xlim(3.5, 7.);
#plt.ylim(3.5, 7.);

plt.scatter(df1.ZOGY_SNR.values,
            df2.ZOGY_SNR.values - df1.ZOGY_SNR.values)
plt.scatter(df1[df1.ZOGY_detected == False].ZOGY_SNR.values,
            df2[df1.ZOGY_detected == False].ZOGY_SNR.values - 
            df1[df1.ZOGY_detected == False].ZOGY_SNR.values, c='r', s=50, alpha=0.2)
#plt.scatter(df1[df2.ZOGY_detected == True].ZOGY_SNR.values,
#            df2[df2.ZOGY_detected == True].ZOGY_SNR.values, c='b', s=50, alpha=0.2)

tmp1 = df1[(df1.ZOGY_detected == True) & (df2.ZOGY_detected == False)]
dit.sizeme(tmp1[tmp1.columns[0:12]])

tmp2 = df2[(df1.ZOGY_detected == True) & (df2.ZOGY_detected == False)]
dit.sizeme(tmp2[tmp2.columns[0:12]])

tmpIm = obj1.D_ZOGY.im - obj2.D_ZOGY.im
print dit.computeClippedImageStats(tmpIm)
tmpVar = obj1.D_ZOGY.var - obj2.D_ZOGY.var
print dit.computeClippedImageStats(tmpVar)

obj2.doPlot([tmp1.inputCentroid_y.values[0], tmp1.inputCentroid_x.values[0], 30], 
            include_Szogy=True, addedImgs=(tmpIm, tmpVar));

im1, im1_psf, im1_var, sig1 = obj2.im1.im, obj2.im1.psf, obj2.im1.var, obj2.im1.sig
im2, im2_psf, im2_var, sig2 = obj2.im2.im, obj2.im2.psf, obj2.im2.var, obj2.im2.sig
F_r = F_n = 1.
padSize = 15 #im1_psf.shape[0]
print im1_psf.shape

print dit.computeClippedImageStats(im1)
#sig1 = dit.computeClippedImageStats(im1).stdev
print dit.computeClippedImageStats(im2)
#sig2 = dit.computeClippedImageStats(im2).stdev

sigR, sigN, P_r_hat, P_n_hat, denom, padded_psf1, padded_psf2 = dit.zogy.ZOGYUtils(im1, im2, im1_psf, im2_psf, sig1, 
                                                                                   sig2, F_r, F_n, padSize=padSize)
print padded_psf1.sum(), padded_psf2.sum()

delta = 0.
K_r_hat = (P_r_hat + delta) / (denom + delta)
K_n_hat = (P_n_hat + delta) / (denom + delta)
K_r = np.fft.ifft2(K_r_hat).real
K_n = np.fft.ifft2(K_n_hat).real

dit.plotImageGrid((padded_psf1, padded_psf2), clim=(-0.01, 0.01))

print K_n.shape
ps = padSize // 2
K_n = K_n[ps:-ps, ps:-ps]
K_r = K_r[ps:-ps, ps:-ps]
print K_n.shape
#K_n /= K_n.sum()
#K_r /= K_r.sum()
dit.plotImageGrid((K_n, K_r), clim=(-0.0001, 0.0001))

import scipy
im1c = scipy.ndimage.filters.convolve(im1, K_n, mode='constant', cval=np.nan)
im2c = scipy.ndimage.filters.convolve(im2, K_r, mode='constant', cval=np.nan)
D = im2c - im1c
D *= np.sqrt(sig1**2. + sig2**2.)

tmpIm = obj1.D_ZOGY.im - D
print dit.computeClippedImageStats(obj1.D_ZOGY.im)
print dit.computeClippedImageStats(D)
print dit.computeClippedImageStats(tmpIm)

obj2.doPlot([tmp1.inputCentroid_y.values[0], tmp1.inputCentroid_x.values[0], 30], include_Szogy=True, 
            addedImgs=(D, tmpIm,));

D2, _ = dit.zogy.performZOGYImageSpace(im1, im2, im1_var, im2_var, im1_psf, im2_psf, sig1, sig2, F_r, F_n, padSize=padSize)
#D2 *= np.sqrt(sig1**2. + sig2**2.)
tmpIm2 = D2 - D
print dit.computeClippedImageStats(D)
print dit.computeClippedImageStats(D2)
print dit.computeClippedImageStats(tmpIm2)

import gzip, cPickle
cPickle.dump((im1, im2, im1_psf, im2_psf), gzip.GzipFile('for_ian.pkl', 'wb'))

cat1 = obj1.D_ZOGY.doDetection(asDF=True, )
cat2 = obj2.D_ZOGY.doDetection(asDF=True, threshold=4.9)
print cat1.shape, cat2.shape
dit.sizeme(cat1.head())
#cat1.columns.values

dit.sizeme(tmp1)

zzz = cat1[(np.abs(cat1.base_PeakCentroid_x.values - tmp1.inputCentroid_x.values[0]) < 1.) &
    (np.abs(cat1.base_PeakCentroid_y.values - tmp1.inputCentroid_y.values[0]) < 1.)]
print zzz.base_GaussianCentroid_flag
print zzz.base_PsfFlux_flag
print zzz.base_PsfFlux_flux, zzz.base_PsfFlux_flux / zzz.base_PsfFlux_fluxSigma
print zzz.base_PeakLikelihoodFlux_flag
print zzz.base_PeakLikelihoodFlux_flux, zzz.base_PeakLikelihoodFlux_flux / zzz.base_PeakLikelihoodFlux_fluxSigma

zzz = cat2[(np.abs(cat2.base_PeakCentroid_x.values - tmp1.inputCentroid_x.values[0]) < 1.) &
    (np.abs(cat2.base_PeakCentroid_y.values - tmp1.inputCentroid_y.values[0]) < 1.)]
print zzz.base_GaussianCentroid_flag
print zzz.base_PsfFlux_flag
print zzz.base_PsfFlux_flux, zzz.base_PsfFlux_flux / zzz.base_PsfFlux_fluxSigma
print zzz.base_PeakLikelihoodFlux_flag
print zzz.base_PeakLikelihoodFlux_flux, zzz.base_PeakLikelihoodFlux_flux / zzz.base_PeakLikelihoodFlux_fluxSigma


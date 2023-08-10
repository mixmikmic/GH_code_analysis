import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

varSourceFlux = 620. * np.sqrt(2.)
n_runs = 1000
inputs = [(f, seed) for f in [varSourceFlux] for seed in np.arange(66, 66+n_runs, 1)]
i = inputs[896]
res = dit.multi.runTest(flux=i[0], seed=i[1], templateNoNoise=False, skyLimited=False, returnObj=True)
print {key+': '+str(res['result'][key]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_decorr']}
testObj = res['obj']

testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True, xaxisIsScienceForcedPhot=True,
                                        skyLimited=True, alpha=0.3);
plt.xlim(600, 1200)
plt.ylim(2, 10.5);
#plt.ylim(5, 10.5);

df = res['df']    
ax = df[df.ZOGY_detected == True].plot.scatter('ZOGY_SNR', 'ALstack_decorr_SNR', c='r', alpha=0.2)
df[df.ZOGY_detected == False].plot.scatter('ZOGY_SNR', 'ALstack_decorr_SNR', c='k', s=10, alpha=0.7, ax=ax)
df[df.ALstack_decorr_detected == True].plot.scatter('ZOGY_SNR', 'ALstack_decorr_SNR', c='b', s=50, alpha=0.2, ax=ax)
#plt.xlim(3.5, 7.);
#plt.ylim(3.5, 7.);

tmp = df[(df.ZOGY_detected == True) & (df.ALstack_decorr_detected == False)]
if tmp.shape[0] <= 0:
    tmp = df
dit.sizeme(tmp)

testObj.doPlot([tmp.inputCentroid_y.values[0], tmp.inputCentroid_x.values[0], 30], include_Szogy=True);

testObj.doPlot([51, 51, 50], include_Szogy=True);

zPSF = testObj.D_ZOGY.psf
adPSF = dit.afw.afwPsfToArray(testObj.ALres.decorrelatedDiffim.getPsf())
aPSF = dit.afw.afwPsfToArray(testObj.ALres.subtractedExposure.getPsf())

print zPSF.shape, adPSF.shape, aPSF.shape
print zPSF.sum(), adPSF.sum(), aPSF.sum()
print dit.psf.computeMoments(zPSF), dit.psf.computeMoments(adPSF), dit.psf.computeMoments(aPSF)
print dit.afw.arrayToAfwPsf(zPSF).computeShape()
print dit.afw.arrayToAfwPsf(adPSF).computeShape()
print dit.afw.arrayToAfwPsf(aPSF).computeShape()

print dit.utils.computeClippedImageStats(adPSF - zPSF)

mk = dit.afw.alPsfMatchingKernelToArray(testObj.ALres.psfMatchingKernel, testObj.ALres.subtractedExposure)
pck = testObj.ALres.decorrelationKernel
print dit.psf.computeMoments(mk)
print dit.psf.computeMoments(pck)
print np.unravel_index(np.argmax(mk), mk.shape)
print np.unravel_index(np.argmax(pck), pck.shape)

dit.plotImageGrid((zPSF, adPSF, aPSF, adPSF-zPSF, mk, pck), clim=(-0.001, 0.001))

zvar = testObj.D_ZOGY.var
advar = testObj.ALres.decorrelatedDiffim.getMaskedImage().getVariance().getArray()
avar = testObj.ALres.subtractedExposure.getMaskedImage().getVariance().getArray()

zim = testObj.D_ZOGY.im
adim = testObj.ALres.decorrelatedDiffim.getMaskedImage().getImage().getArray()
aim = testObj.ALres.subtractedExposure.getMaskedImage().getImage().getArray()

#print dit.computeClippedImageStats(testObj.D_ZOGY.var)
#print dit.computeClippedImageStats(testObj.im1.var)
#print dit.computeClippedImageStats(testObj.im2.var)

print dit.computeClippedImageStats(zim)
print dit.computeClippedImageStats(adim)
print dit.computeClippedImageStats(aim)
print
print dit.computeClippedImageStats(zvar)
print dit.computeClippedImageStats(advar)
print dit.computeClippedImageStats(avar)
print
print dit.computeClippedImageStats(zim/zvar)
print dit.computeClippedImageStats(adim/advar)
print dit.computeClippedImageStats(aim/avar)

print testObj.ALres.task.metadata.get("ALBasisSigGauss")
print testObj.ALres.task.metadata.get("ALBasisDegGauss")

print dit.afw.arrayToAfwPsf(testObj.im1.psf).computeShape().getDeterminantRadius()
print dit.afw.arrayToAfwPsf(testObj.im2.psf).computeShape().getDeterminantRadius()
print np.sqrt(1.99**2 - 1.6**2)




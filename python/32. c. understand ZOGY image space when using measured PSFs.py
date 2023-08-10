import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

import warnings
warnings.filterwarnings('ignore')

import diffimTests as dit

# Normal, ZOGY fourier space
testResults1 = dit.multi.runTest(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=15000,
                                templateNoNoise=False, skyLimited=False,
                                sky=[300., 300.], sourceFluxRange=(500,30000),
                                remeasurePsfs=[True, True], avoidAllOverlaps=0., 
                                variablesAvoidBorder=2.1, returnObjs=True, printErrs=True,
                                zogyImageSpace=False)

# Normal, ZOGY image space
testResults1a = dit.multi.runTest(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=15000,
                                templateNoNoise=False, skyLimited=False,
                                sky=[300., 300.], sourceFluxRange=(500,30000),
                                remeasurePsfs=[True, True], avoidAllOverlaps=0., 
                                variablesAvoidBorder=2.1, returnObjs=True, printErrs=True,
                                zogyImageSpace=True)

# Old, ZOGY fourier space
testResults2 = dit.multi.runTestORIG(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=15000,
                                #templateNoNoise=False, skyLimited=False,
                                sky=[300., 300.], #sourceFluxRange=(500,30000),
                                #remeasurePsfs=[True, True], #avoidAllOverlaps=0., 
                                returnObjs=True, printErrs=True)

print testResults1['resultInputPsf']
del testResults1['resultMeasuredPsf']['sources']
print testResults1['resultMeasuredPsf']

print testResults1a['resultInputPsf']
del testResults1a['resultMeasuredPsf']['sources']
print testResults1a['resultMeasuredPsf']

print testResults2['resultInputPsf']
print testResults2['resultMeasuredPsf']

dit.plotImageGrid((testResults1a['objs'][0].im1.psf, testResults1a['objs'][0].im2.psf,
                   testResults1a['objs'][1].im1.psf, testResults1a['objs'][1].im2.psf,
                   testResults2['objs'][1].im1.psf, testResults2['objs'][1].im2.psf,
                   testResults1a['objs'][1].im1.psf - testResults2['objs'][1].im1.psf,
                   testResults1a['objs'][1].im2.psf - testResults2['objs'][1].im2.psf), clim=(-0.001, 0.001))

dit.plotImageGrid((testResults1['objs'][1].im1.im,), imScale=5, clim=(-10, 1000))

print dit.computeClippedImageStats(testResults1['objs'][1].D_ZOGY.im - testResults1a['objs'][1].D_ZOGY.im)
dit.plotImageGrid((testResults1['objs'][1].D_ZOGY.im - testResults1a['objs'][1].D_ZOGY.im,), imScale=5)

testResults1['objs'][0].reset()
print testResults1['objs'][0].runTest(zogyImageSpace=False);
testResults1a['objs'][0].reset()
print testResults1a['objs'][0].runTest(zogyImageSpace=True);

print dit.computeClippedImageStats(testResults1['objs'][0].D_ZOGY.im - testResults1a['objs'][0].D_ZOGY.im)
dit.plotImageGrid((testResults1['objs'][0].D_ZOGY.im - testResults1a['objs'][0].D_ZOGY.im,), imScale=5)

# Normal, ZOGY fourier space, less noisy template
testResults3 = dit.multi.runTest(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=15000,
                                templateNoNoise=False, skyLimited=False,
                                sky=[30., 300.], sourceFluxRange=(500,30000),
                                remeasurePsfs=[True, True], avoidAllOverlaps=0., 
                                variablesAvoidBorder=2.1, returnObjs=True, printErrs=True,
                                zogyImageSpace=False)

# Noisy template (sky = 300)
print testResults1['resultInputPsf']
#del testResults1['resultMeasuredPsf']['sources']
print testResults1['resultMeasuredPsf']

# Less noisy template (sky = 30), ZOGY not image-space
print testResults3['resultInputPsf']
del testResults3['resultMeasuredPsf']['sources']
print testResults3['resultMeasuredPsf']

dit.plotImageGrid((testResults1['objs'][0].im1.psf, testResults1['objs'][0].im2.psf,
                   testResults1['objs'][1].im1.psf, testResults1['objs'][1].im2.psf,
                   testResults3['objs'][0].im1.psf, testResults3['objs'][0].im2.psf,
                   testResults3['objs'][1].im1.psf, testResults3['objs'][1].im2.psf,
                   testResults1['objs'][1].im1.psf - testResults3['objs'][1].im1.psf,
                   testResults1['objs'][1].im2.psf - testResults3['objs'][1].im2.psf), clim=(-0.001, 0.001))

print dit.computeClippedImageStats(testResults1['objs'][1].D_ZOGY.im)
print dit.computeClippedImageStats(testResults3['objs'][1].D_ZOGY.im)
print dit.computeClippedImageStats(testResults1['objs'][1].D_ZOGY.var)
print dit.computeClippedImageStats(testResults3['objs'][1].D_ZOGY.var)
#print dit.computeClippedImageStats(testResults1['objs'][1].D_ZOGY.im - testResults3['objs'][1].D_ZOGY.im)
print np.sqrt(300+300), np.sqrt(300+30)
#dit.plotImageGrid((testResults1['objs'][1].D_ZOGY.im - testResults3['objs'][1].D_ZOGY.im,), imScale=5)

# Normal, ZOGY fourier space, less noisy template, but dont re-fit its PSF
testResults3a = dit.multi.runTest(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=15000,
                                templateNoNoise=False, skyLimited=False,
                                sky=[30., 300.], sourceFluxRange=(500,30000),
                                remeasurePsfs=[False, True], avoidAllOverlaps=0., 
                                variablesAvoidBorder=2.1, returnObjs=True, printErrs=True,
                                zogyImageSpace=False)

# Less noisy template (sky = 30), ZOGY not image-space, don't re-fit template PSF
print testResults3a['resultInputPsf']
del testResults3a['resultMeasuredPsf']['sources']
print testResults3a['resultMeasuredPsf']

# Compate it to the same but don't refit EITHER psf.
testResults3b = dit.multi.runTest(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=15000,
                                templateNoNoise=False, skyLimited=False,
                                sky=[30., 300.], sourceFluxRange=(500,30000),
                                remeasurePsfs=[False, False], avoidAllOverlaps=0., 
                                variablesAvoidBorder=2.1, returnObjs=True, printErrs=True,
                                zogyImageSpace=False)

print testResults3b['resultInputPsf']
#del testResults3b['resultMeasuredPsf']['sources']
print testResults3b['resultMeasuredPsf']

print dit.computeClippedImageStats(testResults1['objs'][1].D_ZOGY.im)
print dit.computeClippedImageStats(testResults3['objs'][1].D_ZOGY.im)
print dit.computeClippedImageStats(testResults3a['objs'][1].D_ZOGY.im)
print dit.computeClippedImageStats(testResults1['objs'][1].D_ZOGY.var)
print dit.computeClippedImageStats(testResults3['objs'][1].D_ZOGY.var)
print dit.computeClippedImageStats(testResults3a['objs'][1].D_ZOGY.var)
#print dit.computeClippedImageStats(testResults1['objs'][1].D_ZOGY.im - testResults3['objs'][1].D_ZOGY.im)
print np.sqrt(300+300), np.sqrt(300+30)
#dit.plotImageGrid((testResults1['objs'][1].D_ZOGY.im - testResults3['objs'][1].D_ZOGY.im,), imScale=5)

print dit.computeClippedImageStats(testResults3b['objs'][0].D_ZOGY.im - testResults3a['objs'][1].D_ZOGY.im)
dit.plotImageGrid((testResults3b['objs'][0].D_ZOGY.im - testResults3a['objs'][1].D_ZOGY.im,), imScale=5)




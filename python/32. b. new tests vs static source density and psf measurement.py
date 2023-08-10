import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

import warnings
warnings.filterwarnings('ignore')

testResults2 = dit.multi.runTest(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=4000,
                                templateNoNoise=False, skyLimited=False,
                                sky=[300., 300.], sourceFluxRange=(500,30000),
                                remeasurePsfs=[True, True], avoidAllOverlaps=25., 
                                returnObj=True, printErrs=True)

testResults2['resultInputPsf']

#del testResults2['sources']
testResults2['resultMeasuredPsf']

n_runs = 5 # 10
ns = np.append(np.insert(np.arange(500, 5001, 250), 0, [50, 100, 250]), [7500, 10000, 15000])
ns = ns[::2]
testResults2 = dit.multi.runMultiDiffimTests(varSourceFlux=620.*np.sqrt(2.), 
                                             n_varSources=50, nStaticSources=ns,
                                             templateNoNoise=False, skyLimited=False,
                                             sky=[300., 300.], sourceFluxRange=(500,30000),
                                             avoidAllOverlaps=25., zogyImageSpace=False,
                                             n_runs=n_runs, remeasurePsfs=[True, True])

tr = testResults2
tr = [t for t in tr if t is not None and t['resultMeasuredPsf']]
dit.multi.plotResults(tr, resultKey='resultMeasuredPsf');









import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np

import warnings
warnings.filterwarnings('ignore')

#import diffimTests_OLD as ditOLD
#reload(ditOLD);
import diffimTests_OLD_multi as ditMultiOLD
reload(ditMultiOLD);

testResults = ditMultiOLD.runTest(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=1000,
                                #templateNoNoise=False, skyLimited=False,
                                sky=300., sourceFluxRange=(500,30000),
                                remeasurePsfs=[True, True], #avoidAllOverlaps=0., 
                                returnObj=True, printErrs=True)

testResults['resultInputPsf']

testResults['resultMeasuredPsf']





testResults2 = ditMultiOLD.runTestORIG(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=1000,
                                #templateNoNoise=False, skyLimited=False,
                                sky=300.) #, #sourceFluxRange=(500,30000),
                                #remeasurePsfs=[True, True], #avoidAllOverlaps=0., 
                                #returnObj=True, printErrs=True)

#del testResults2['inputPsf1'], testResults2['inputPsf2'], testResults2['psf1'], testResults2['psf2']
print testResults2['diffimResInputPsf']
print testResults2['diffimResMeasuredPsf']





testResults3 = ditMultiOLD.runTestORIG(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=1000,
                                #templateNoNoise=False, skyLimited=False,
                                sky=300., #sourceFluxRange=(500,30000),
                                #remeasurePsfs=[True, True], #avoidAllOverlaps=0., 
                                #returnObj=True, 
                                       printErrs=True)

print testResults3['diffimResInputPsf']
print testResults3['diffimResMeasuredPsf']

testResults3 = ditMultiOLD.runTestORIG(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=4000,
                                #templateNoNoise=False, skyLimited=False,
                                sky=300., #sourceFluxRange=(500,30000),
                                #remeasurePsfs=[True, True], #avoidAllOverlaps=0., 
                                #returnObj=True, 
                                       returnObjs=True, printErrs=True)

print testResults3['diffimResInputPsf']
print testResults3['diffimResMeasuredPsf']

ditOLD.plotImageGrid((testResults3['objs'][1].im1.psf, testResults3['objs'][1].im2.psf), clim=(-0.001, 0.001))





testResults4 = ditMultiOLD.runTestORIG(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=4000,
                                #templateNoNoise=False, skyLimited=False,
                                sky=300., #sourceFluxRange=(500,30000),
                                #remeasurePsfs=[True, True], #avoidAllOverlaps=0., 
                                #returnObj=True, 
                                       returnObjs=True, printErrs=True)

print testResults4['diffimResInputPsf']
print testResults4['diffimResMeasuredPsf']

ditOLD.plotImageGrid((testResults4['objs'][1].im1.psf, testResults4['objs'][1].im2.psf), clim=(-0.001, 0.001))





testResults5 = ditMultiOLD.runTestORIG(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=4000,
                                #templateNoNoise=False, skyLimited=False,
                                sky=300., #sourceFluxRange=(500,30000),
                                #remeasurePsfs=[True, True], #avoidAllOverlaps=0., 
                                #returnObj=True, 
                                       returnObjs=True, printErrs=True)

print testResults5['diffimResInputPsf']
print testResults5['diffimResMeasuredPsf']







testResults6 = ditMultiOLD.runTestORIG(flux=620.*np.sqrt(2.), n_varSources=50, n_sources=4000,
                                #templateNoNoise=False, skyLimited=False,
                                sky=300., #sourceFluxRange=(500,30000),
                                #remeasurePsfs=[True, True], #avoidAllOverlaps=0., 
                                #returnObj=True, 
                                       returnObjs=True, printErrs=True)

print testResults6['diffimResInputPsf']
print testResults6['diffimResMeasuredPsf']





n_runs = 5 # 10
ns = np.append(np.insert(np.arange(500, 5001, 250), 0, [50, 100, 250]), [7500, 10000, 15000])
ns = ns[::2]
testResults4 = ditMultiOLD.runMultiDiffimTests(varSourceFlux=620.*np.sqrt(2.), 
                                             n_varSources=50, nStaticSources=ns,
                                             #templateNoNoise=False, skyLimited=False,
                                             sky=[300., 300.], #sourceFluxRange=(500,30000),
                                             #avoidAllOverlaps=0.,
                                             n_runs=n_runs) #, remeasurePsfs=[True, True])

import diffimTests as dit
dit.multi.plotResults(testResults4, resultKey='diffimResInputPsf');

dit.multi.plotResults(testResults4, resultKey='diffimResMeasuredPsf');

for tr in testResults4:
    tr['n_sources'] = tr['nSources']
plotMeasuredPsfsResults(testResults4, resultKey='resultPsfRms', methods=['ALstack_decorr', 'ZOGY']);

for tr in testResults4:
    tr['n_sources'] = tr['nSources']
plotMeasuredPsfsResults(testResults4, resultKey='diffimResInputPsf', methods=['ALstack_decorr', 'ZOGY']);

for tr in testResults4:
    tr['n_sources'] = tr['nSources']
plotMeasuredPsfsResults(testResults4, resultKey='diffimResMeasuredPsf', methods=['ALstack_decorr', 'ZOGY']);











import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import diffimTests as dit

n_runs = 10
ns = np.append(np.insert(np.arange(500, 5001, 250), 0, [50, 100, 250]), [7500, 10000, 15000])
#ns = ns[::2]
testResults5 = dit.multi.runMultiDiffimTestsORIG(varSourceFlux=620.*np.sqrt(2.), 
                                             n_varSources=50, nStaticSources=ns,
                                             #templateNoNoise=False, skyLimited=False,
                                             sky=[300., 300.], #sourceFluxRange=(500,30000),
                                             #avoidAllOverlaps=0.,
                                             n_runs=n_runs) #, remeasurePsfs=[True, True])

dit.multi.plotMeasuredPsfsResults(testResults5, resultKey='resultPsfRms', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults5, resultKey='diffimResInputPsf', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults5, resultKey='diffimResMeasuredPsf', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults5, resultKey='diffimResMeasuredPsf', methods=['ALstack', 'ZOGY']);

n_runs = 50
ns = np.append(np.insert(np.arange(500, 5001, 250), 0, [50, 100, 250]), [7500, 10000, 15000, 20000, 50000])
#ns = ns[::2]
testResults6 = dit.multi.runMultiDiffimTestsORIG(varSourceFlux=620.*np.sqrt(2.), 
                                             n_varSources=50, nStaticSources=ns,
                                             #templateNoNoise=False, skyLimited=False,
                                             sky=[300., 300.], #sourceFluxRange=(500,30000),
                                             #avoidAllOverlaps=0.,
                                             n_runs=n_runs) #, remeasurePsfs=[True, True])

dit.multi.plotMeasuredPsfsResults(testResults6, resultKey='resultPsfRms', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults6, resultKey='diffimResInputPsf', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults6, resultKey='diffimResMeasuredPsf', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults6, resultKey='diffimResMeasuredPsf', methods=['ALstack', 'ZOGY']);

dit.dumpObjects((testResults5, testResults6), '32. b. new tests vs static source density and psf measurement')











import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

import warnings
warnings.filterwarnings('ignore')

n_runs = 20 #10
ns = np.append(np.insert(np.arange(500, 5001, 250), 0, [50, 100, 250]), [7500, 10000, 15000])
#ns = ns[::2]
testResults6 = dit.multi.runMultiDiffimTests(varSourceFlux=620.*np.sqrt(2.), 
                                             n_varSources=50, nStaticSources=ns,
                                             templateNoNoise=False, skyLimited=False,
                                             sky=[300., 300.], sourceFluxRange=(500,30000),
                                             avoidAllOverlaps=25., zogyImageSpace=False,
                                             n_runs=n_runs, remeasurePsfs=[True, True],
                                            printErrs=True)

dit.multi.plotMeasuredPsfsResults(testResults6, resultKey='resultPsfRms', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults6, resultKey='resultInputPsf', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults6, resultKey='resultMeasuredPsf', methods=['ALstack_decorr', 'ZOGY']);

dit.multi.plotMeasuredPsfsResults(testResults6, resultKey='resultMeasuredPsf', methods=['ALstack', 'ZOGY']);




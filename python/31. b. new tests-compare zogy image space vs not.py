import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

import warnings
warnings.filterwarnings('ignore')

n_runs = 100
fluxes = np.repeat(620., 50)
testResults1 = dit.multi.runMultiDiffimTests(varSourceFlux=fluxes, n_runs=n_runs,
                                            remeasurePsfs=[False, False])
testResults2 = dit.multi.runMultiDiffimTests(varSourceFlux=fluxes, n_runs=n_runs,
                                            remeasurePsfs=[False, False], zogyImageSpace=True)
#testResults2 = dit.multi.runMultiDiffimTests(varSourceFlux=fluxes, n_runs=n_runs, 
#                                             templateNoNoise=False)
#testResults3 = dit.multi.runMultiDiffimTests(varSourceFlux=fluxes, n_runs=n_runs, 
#                                             templateNoNoise=False, skyLimited=False)

dit.dumpObjects((testResults1, testResults2), "tmp3_pkl")

testResults1, testResults2 = dit.loadObjects('tmp3_pkl')

dit.multi.plotResults(testResults1, title='Noise-free template; sky-limited');

dit.multi.plotResults(testResults2, title='Noise-free template; sky-limited');

res = dit.multi.runTest(np.linspace(200., 2000., 50), returnObj=True)

testObj = res['obj']
testObj.doPlotWithDetectionsHighlighted(runTestResult=res, addPresub=True, divideByInput=False)
plt.xlim(0, 2000)

testObj = res['obj']
testObj.doPlotWithDetectionsHighlighted(runTestResult=res, addPresub=True, divideByInput=True)
plt.xlim(0, 2000)




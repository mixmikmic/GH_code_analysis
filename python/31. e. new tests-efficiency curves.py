import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

import warnings
warnings.filterwarnings('ignore')

n_runs = 100
fluxes = np.linspace(200., 2000., 40)
testResults1 = dit.multi.runMultiDiffimTests(varSourceFlux=fluxes, n_runs=n_runs)
testResults2 = dit.multi.runMultiDiffimTests(varSourceFlux=fluxes, n_runs=n_runs, 
                                             templateNoNoise=False)
testResults3 = dit.multi.runMultiDiffimTests(varSourceFlux=fluxes, n_runs=n_runs, 
                                             templateNoNoise=False, skyLimited=False)

dit.dumpObjects((testResults1, testResults2, testResults3), "tmp2_pkl")

testResults1, testResults2, testResults3 = dit.loadObjects('tmp2_pkl')

dit.multi.plotResults(testResults1, title='Noise-free template; sky-limited');

dit.multi.plotResults(testResults1, title='Noise-free template; sky-limited', asHist=True, doPrint=False);

dit.multi.plotSnrResults(testResults1, title='Noise-free template; sky-limited');

dit.multi.plotSnrResults(testResults2, title='Noise-free template; sky-limited');

dit.multi.plotSnrResults(testResults3, title='Noise-free template; sky-limited');




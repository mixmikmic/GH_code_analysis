import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

import warnings
warnings.filterwarnings('ignore')

n_runs = 100
#psf2 = [2.2, 2.4]
testResults1 = dit.multi.runMultiDiffimTests(varSourceFlux=620., n_runs=n_runs) #, psf2=psf2)
testResults2 = dit.multi.runMultiDiffimTests(varSourceFlux=620.*np.sqrt(2.), n_runs=n_runs, 
                                             templateNoNoise=False) #, psf2=psf2)
testResults3 = dit.multi.runMultiDiffimTests(varSourceFlux=620.*np.sqrt(2.), n_runs=n_runs, 
                                             templateNoNoise=False, skyLimited=False) #, psf2=psf2)

dit.dumpObjects((testResults1, testResults2, testResults3), 'tmp_pkl')

testResults1, testResults2, testResults3 = dit.loadObjects('tmp_pkl')
#testResults1, testResults2, testResults3 = dit.loadObjects('tmp_pkl_1000runs')

dit.multi.plotResults(testResults1, title='Noise-free template; sky-limited');

dit.multi.plotResults(testResults1, title='Noise-free template; sky-limited', asHist=True, doPrint=False);

dit.multi.plotSnrResults(testResults1, title='Noise-free template; sky-limited');

dit.multi.plotResults(testResults2, title='Noisy template; sky-limited');

dit.multi.plotSnrResults(testResults2, title='Noisy template; sky-limited');

dit.multi.plotResults(testResults3, title='Noisy template; not sky-limited');

dit.multi.plotSnrResults(testResults3, title='Noisy template; not sky-limited');

TP, FP, FN = dit.multi.plotResults(testResults2, actuallyPlot=False, doPrint=False)
TP.ALstack_decorr += np.random.uniform(-0.2, 0.2, TP.shape[0])
TP.ZOGY += np.random.uniform(-0.2, 0.2, TP.shape[0])
TP.plot.scatter('ALstack_decorr', 'ZOGY')

TP, FP, FN = dit.multi.plotResults(testResults2, actuallyPlot=False, doPrint=False)
print np.max(TP.ALstack_decorr/TP.ZOGY)
print 1./np.min(TP.ALstack_decorr/TP.ZOGY)
TP[TP.ZOGY/TP.ALstack_decorr > 1.26]




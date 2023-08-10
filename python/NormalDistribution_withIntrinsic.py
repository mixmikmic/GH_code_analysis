# plot within the notebook
get_ipython().magic('matplotlib inline')
import warnings
# No annoying warnings
warnings.filterwarnings('ignore')
# Because we always need that
import numpy as np
import matplotlib.pyplot as mpl

errors= np.random.rand(200)/0.5 +0.2
data = np.random.normal(loc=3, scale=0.5, size=200) * (1 + np.random.normal(loc=0, scale=errors))

fig = mpl.figure(figsize=[10,5])
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
prop = dict( histtype="step", fill=True, fc=mpl.cm.Blues(0.5,0.3), ec="k", lw=2)
ax1.hist(data, **prop)
ax2.hist(errors,**prop)

print "mean: ", np.mean(data)
print "std: ", np.std(data)
print " => error on the mean: ",np.std(data)/np.sqrt(len(data))
print " => error on the std: ",np.std(data)/np.sqrt(2*len(data))

import modefit
reload(modefit) ; reload(modefit.fitter) ; reload(modefit.fitter.unimodal)

normfit = modefit.normalfit(data,errors)

normfit.fit(mean_guess=2, mean_boundaries=[-10,10], sigma_guess=3, sigma_boundaries=[0,10])




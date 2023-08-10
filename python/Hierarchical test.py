import numpy as np
scores = np.loadtxt('Data/diffNbcHnb.csv', delimiter=',')
names = ("HNB", "NBC")
print(scores)

import bayesiantests as bt    
rope=0.01 #we consider two classifers equivalent when the difference of accuracy is less that 1%
rho=1/10 #we are performing 10 folds, 10 runs cross-validation
pleft, prope, pright=bt.hierarchical(scores,rope,rho)

pl, pe, pr=bt.hierarchical(scores,rope,rho, verbose=True, names=names)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

samples=bt.hierarchical_MC(scores,rope,rho, names=names)

#plt.rcParams['figure.facecolor'] = 'black'

fig = bt.plot_posterior(samples,names)
plt.savefig('triangle_hierarchical.png',facecolor="black")
plt.show()


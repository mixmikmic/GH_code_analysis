import numpy as np
scores = np.loadtxt('Data/accuracy_nbc_aode.csv', delimiter=',', skiprows=1, usecols=(1, 2))
names = ("NBC", "AODE")

import bayesiantests as bt
left, within, right = bt.signtest(scores, rope=0.01)
print(left, within, right)

left, within, right = bt.signtest(scores, rope=0.01, verbose=True, names=names)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'black'

samples = bt.signtest_MC(scores, rope=0.01)

fig = bt.plot_posterior(samples,names)
plt.savefig('triangle.png',facecolor="black")
plt.show()

samples = bt.signtest_MC(scores, rope=0.01,  prior_strength=1, prior_place=bt.LEFT)
fig = bt.plot_posterior(samples,names)
plt.show()

samples = bt.signtest_MC(scores, rope=0.01,  prior_strength=1, prior_place=bt.RIGHT)
fig = bt.plot_posterior(samples,names)
plt.show()

samples = bt.signtest_MC(scores, rope=0.01,  prior_strength=10, prior_place=bt.LEFT)
fig = bt.plot_posterior(samples,names)
plt.show()




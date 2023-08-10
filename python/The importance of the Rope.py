import numpy as np
scores = np.loadtxt('Data/accuracy_j48_j48gr.csv', delimiter=',', skiprows=1, usecols=(1, 2))
names = ("J48", "J48gr")

import bayesiantests as bt
left, within, right = bt.signtest(scores, rope=0.01,verbose=True,names=names)

left, within, right = bt.signtest(scores, rope=0.001,verbose=True,names=names)

left, within, right = bt.signtest(scores, rope=0.0001,verbose=True,names=names)

left, within, right = bt.signtest(scores, rope=0.0001,prior_place=bt.RIGHT,verbose=True,names=names)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
left=np.zeros((10,1))
within=np.zeros((10,1))
right=np.zeros((10,1))
for i in range(9,-1,-1):
    left[i], within[i], right[i] = bt.signtest(scores, rope=0.001/2**i,names=names)
plt.plot(0.001/(2**np.arange(0,10,1)),within)
plt.plot(0.001/(2**np.arange(0,10,1)),left)
plt.plot(0.001/(2**np.arange(0,10,1)),right)
plt.legend(('rope','left','right'))
plt.xlabel('Rope width')
plt.ylabel('Probability')

left, within, right = bt.signrank(scores, rope=0.001,verbose=True,names=names)

left, within, right = bt.signrank(scores, rope=0.0001,verbose=True,names=names)




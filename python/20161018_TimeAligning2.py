import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#Construct the Markov Chain array. For ease of visualization, all cells
#have two decimal places.
mC = np.array([[0.03,0.97,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                [0.17,0.17,0.66,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                [0.00,0.11,0.42,0.47,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                [0.00,0.00,0.06,0.70,0.24,0.00,0.00,0.00,0.00,0.00,0.00],
                [0.00,0.00,0.00,0.03,0.87,0.10,0.00,0.00,0.00,0.00,0.00],
                [0.00,0.00,0.00,0.00,0.03,0.94,0.03,0.00,0.00,0.00,0.00],
                [0.00,0.00,0.00,0.00,0.00,0.22,0.73,0.05,0.00,0.00,0.00],
                [0.00,0.00,0.00,0.00,0.00,0.00,0.38,0.52,0.10,0.00,0.00],
                [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.53,0.33,0.14,0.00],
                [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.66,0.18,0.16],
                [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.91,0.09]])
#Plot a heatmap of the Markov Chain
plt.figure(0)
hm = plt.pcolor(mC,cmap='Blues');
plt.colorbar(hm);

import random
#Finding the cumulative sum means that we can draw a random number 
#between 0 to 1, iterate through each state, and check if the 
#random number is less than the cell value to determine which 
#state transition occurs.
mCS = np.cumsum(mC,axis=1)
#Initialization
curState = 5
trueStates = [5]
#Iterate over a total of 100,000 time steps
for ii in range(100000):    
    rn = random.random()
    #Iterate all possible states to find the state transition
    for jj in range(11):
        if (rn < mCS[curState,jj]):
            curState = jj
            break
    #Save the current state to an array      
    trueStates.append(curState)
#Now, work backwards to "compress" the data so it matches the 
#historian algorithm
prevState = 5   
rawStates = [5]
for ii in range(1,len(trueStates)):
    ts = trueStates[ii]
    if (prevState == ts):
        rawStates.append(None)
    else:
        rawStates[ii-1] = prevState
        rawStates.append(ts)
        prevState = ts

from scipy.stats import kurtosis
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#Remove the None values from the rawStates
rawStatesR = [ii for ii in rawStates if ii is not None]
#Generate text boxes containing the mean and std
txtT = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%    (np.mean(trueStates),np.median(trueStates),np.std(trueStates))
txtR = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%    (np.mean(rawStatesR),np.median(rawStatesR),np.std(rawStatesR))
#Plot using matplotlib
plt.figure(1, figsize=[10,8])
ax = plt.subplot(1,2,1)
plt.hist(trueStates,normed=1);
ax.text(0.05, 0.95, txtT, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.title('True States')
plt.axis([0,11,0,0.7])
ax = plt.subplot(1,2,2)
plt.hist(rawStatesR,normed=1);
ax.text(0.05, 0.95, txtR, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.title('Raw States')
plt.axis([0,11,0,0.7]);


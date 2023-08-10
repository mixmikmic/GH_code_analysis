import numpy as np  
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().magic('matplotlib inline')

#two populations: 
np.random.seed(1) #ensure we get the same results every time 
y1 = np.random.normal(5.8, .7, 200) #normally distributed data centered on 5.8ft, variance of .7ft, 200 samples
y2 = np.random.normal(6.0, .7, 200) #normally distributed data centered on 6ft, variance of .7ft, 200 samples

#plot histograms of data 
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1,1,1)
ax1.hist(y1, bins=15)
ax1.hist(y2, bins=15)

plt.ylabel('frequency');
plt.xlabel('height');
b_patch = mpatches.Patch(color='blue', label='hometown heights');
r_patch = mpatches.Patch(color='orange', label='workplace heights');
plt.legend(handles=[b_patch, r_patch])
plt.show()

#run t-test, compute p and t values 
t_val, p_val = ttest_ind(y1, y2)
print 't_test results:'
print 't-value', t_val
print 'p-value', p_val




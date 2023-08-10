import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

mean=0
std=1
x_value = 1.5

#create x values by feeding in mean and std into ppf
x= np.linspace(st.norm.ppf(0.001,mean,std), st.norm.ppf(0.999,mean,std), 100)

#create y values by feeding in x values, mean, and std into pdf
y=st.norm.pdf(x,mean,std)

#plt distribution
fig, ax = plt.subplots(1, 1)
ax.plot(x, y,'b-', lw=3, alpha=0.7, label='Norm PDF')

#put dotted line at mean and x_value, and fill the area under the curve!
ax.axvline(mean, color='b', linestyle='--', label='Mean: '+str(round(mean,2)))
ax.axvline(x_value, color='r', linestyle='-', label='z-value')
ax.fill_between(x,0,y, where= ( np.logical_and(x>=0, x<=x_value)), facecolor='green', alpha=0.4)

#put in titles!
plt.suptitle('Normal Distribution (mean='+str(mean)+',std='+str(std)+')')
plt.title('Oooo nice plot!')
plt.xlabel('Quanatiles')
plt.ylabel('Density')
plt.legend(loc='upper right')

plt.show()


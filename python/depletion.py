get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams["font.size"] = 18

import math
def n_decay(t, n_initial=100, half_life=1):
    """This function describes the decay of an isotope"""
    lam = math.log(2)/half_life
    return n_initial*math.exp(-lam*t)


# This code plots the decay of an isotope
import numpy as np
y = np.arange(24)
x = np.arange(24)
for t in range(0,24):
    x[t] = t
    y[t] = n_decay(t)
    
# creates a figure and axes with matplotlib
fig, ax = plt.subplots()
scatter = plt.scatter(x, y, color='blue', s=y*20, alpha=0.4)    
ax.plot(x, y, color='red')    

# adds labels to the plot
ax.set_ylabel('N_i(t)')
ax.set_xlabel('Time')
ax.set_title('N_i')

# adds tooltips
import mpld3
labels = ['{0}% remaining'.format(i) for i in y]
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.display()




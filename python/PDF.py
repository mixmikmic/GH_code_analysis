get_ipython().magic('matplotlib inline')
from __future__ import division
def F(x):
    if x<0:
        return 0
    if x<0.5:
        return x
    if x>=0.5 and x<=1:
        return x/2+0.5
    return 1
    
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-2,2, num=100)
fx =[F(i) for i in x]
pos = np.where((x<=0.51) & (x>=0.49))[0]

x[pos] = np.nan
fx[pos] = np.nan
plt.plot(x,fx)



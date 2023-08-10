get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

plt.plot([1,4,9,16,25])

plt.plot(np.sin([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]))

x = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6];
y = np.sin(x);
plt.plot(x,y)


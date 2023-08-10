import matplotlib.pyplot as plt
#just to plot graph in ipython notebook
get_ipython().magic('matplotlib inline')

import numpy as np

x = np.linspace(0,10,100)

plt.plot(x,np.sin(x), '.')
plt.show()

plt.plot(x,np.cos(x),'-.')

x = np.arange(10000)
y = np.square(x) 

plt.plot(x,y,'-')
plt.show()

plt.subplot(2,1,1)
plt.plot(x,np.sin(x))

plt.subplot(2,1,2)
plt.plot(x,np.cos(x))

x = np.linspace(0,10,100)
plt.subplot(2,2,1)
plt.plot(x,np.sin(x))

plt.subplot(2,2,2)
plt.plot(x,np.cos(x))

plt.subplot(2,2,3)
plt.plot(x,np.cos(x))

plt.subplot(2,2,4)
plt.plot(x,np.cos(x))

plt.subplot(1,2,1)
plt.plot(x,np.sin(x))

plt.subplot(1,2,2)
plt.plot(x,np.cos(x))

fig, ax =  plt.subplots(3,2)
ax[0,0].plot(x,np.sin(x))
ax[0,1].plot(x,np.sin(x))
fig.show()



fig




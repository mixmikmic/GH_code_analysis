

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

import numpy as np

X = np.linspace(0,1,1000)

Y = np.square(X) + 5*X + 7

plt.plot(X,Y,'-')
plt.show()

plt.plot(X, np.sin(X),'-k')
plt.plot(X, np.tan(X), '-r')

X = np.arange(100)

plt.plot(X,np.tan(X))

plt.plot(np.arange(10),np.sinh(np.arange(10)))






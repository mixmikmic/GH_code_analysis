get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,10)
a = 0.05

plt.plot(t,np.exp(0.2*t))
plt.plot(t,np.exp(0*t))
plt.plot(t,np.exp(-0.2*t))

plt.legend(['a = 0.2','a = 0.0','a = -0.2'],loc = 'upper left')
plt.xlabel('Time')
plt.title('Solutions x(t) = exp(a*t)')




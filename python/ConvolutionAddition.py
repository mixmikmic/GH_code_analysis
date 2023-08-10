import math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(12,12))

# First Gaussian
g1 = [np.random.normal(-2, 1) for x in range(1000000)]
v1, e1 = np.histogram(g1, bins=1000, range=(-5,5), density=True)
sup = [(e1[i] + e1[i+1]) / 2 for i in range(len(e1)-1)]
plt.plot(sup, v1, color='green', alpha=0.5)    
    
# Second Gaussian
g2 = [np.random.normal(2, 1) for x in range(1000000)]
v2, _ = np.histogram(g2, bins=1000, range=(-5,5), density=True)
plt.plot(sup, v2, color='green', alpha=0.5)

# Convolution 
v = np.convolve(v1, v2, mode='same')

# Normalization
norm = 1.0 / math.sqrt(8000)
v = [x * norm for x in v]
plt.plot(sup, v, color='red')

plt.show()




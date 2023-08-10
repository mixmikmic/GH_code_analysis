import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#%matplotlib qt

from pylab import rcParams
rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

spending = [2538.21,2160.6,1211.96,1173.38,1714.89,1541.97,1475.23,3519.89,3759.7,1380.6,2955.24,1529.63,1512.34,1491.18,1471.51,786.05,1905.11,3358.46,2516.27,2595.3,2724.17,2836.07,4111.47,3505.72,4552.18,5781.79,3544.94,3151.69,3397.3,7685.04,3591.86,3143.16,4294.07,2879.07,1770.49,1774.74,1519.68,2079.18,2006.58,3513.78,1380.16,2848.3,2054.01,4471.03,2483.99,3173.29,5273.45,-303.76,4340.43,1750.23,2843.33,3403.78,4829.28,2811.32,3649.22,2269.74,2597.96,3489.25,3060.98,1094.1,3644.47,3934.78,406.01,2722.86,3692.34,2604.96,4306.01,3891.58,3181.66,3411.69,2761.02,1614.47,4277.09,3337.82,3162.37,4290.53,3289.63,2839.71]

plt.plot(spending)
months = np.arange(len(spending))
ps = np.polyfit(months, spending, 1)
plt.plot(months, np.sum([p*months**(1-i) for i,p in enumerate(ps)], axis=0))

ps

13*75

spending[-1] - spending[0]




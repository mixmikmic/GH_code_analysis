get_ipython().magic('pylab inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, rc=None)

x = np.linspace(-10,10,1000)
cdf = 1/(1+np.exp(-x))

plt.figure(figsize = (8,8))
plt.plot(x, cdf, 'b')
plt.xlabel('Support', fontsize = 16)
plt.ylabel('Probability', fontsize = 16)
#plt.ylim((0,.3))
plt.title('Logistic CDF', fontsize = 20)
plt.axvline(0, color='r', ls='--', lw=2.0)


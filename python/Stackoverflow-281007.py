get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=2)

mu1 = 1
sigma1 = 2

mu2 = 2
sigma2 = 3

np.random.seed(42)
X = np.random.normal(mu1, sigma1, 1000)
Y = np.random.normal(mu2, sigma2, 1000)


# P(X|X>Y)
P_X_XgY = X[X>Y]

P_X_XlY = X[X<Y]

denom = 1-sp.stats.norm.cdf((mu2-mu1)/np.sqrt(sigma1**2+sigma2**2))

count, bins, ignored = plt.hist(P_X_XgY, 30, normed=True)
plt.plot(bins, 1/(sigma1 * np.sqrt(2 * np.pi)) * (sp.stats.norm.cdf((bins-mu2)/sigma2)/denom) * np.exp( - (bins - mu1)**2 / (2 * sigma1**2) ), linewidth=2, color='r')
plt.title('$P(X|X>Y)$')
plt.legend('')

denom = sp.stats.norm.cdf((mu2-mu1)/np.sqrt(sigma1**2+sigma2**2))

count, bins, ignored = plt.hist(P_X_XlY, 30, normed=True)
plt.plot(bins, 1/(sigma1 * np.sqrt(2 * np.pi)) * ((1-sp.stats.norm.cdf((bins-mu2)/sigma2))/denom) * np.exp( - (bins - mu1)**2 / (2 * sigma1**2) ), linewidth=2, color='r')
plt.title('$P(X|X<Y)$')


# %load std_ipython_import.txt
import pandas as pd
import scipy as spy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.mixture import GMM

from matplotlib.colors import LogNorm

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)
pd.set_option('expand_frame_repr', True)

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_formats = {'retina',}")

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

data = pd.read_csv('data/SAheart.data', sep=',', decimal='.', usecols=np.arange(1,11))
data.info()

mixm = GMM(n_components=1, covariance_type='tied')
x = np.linspace(data.age.min(), data.age.max()).reshape(-1,1)

mixm.fit(data[data['chd'] == 0]['age'].reshape(-1,1))
no_chd = np.exp(mixm.score(x))

mixm.fit(data[data['chd'] == 1]['age'].reshape(-1,1))
chd = np.exp(mixm.score(x))

mixm.set_params(n_components=2)
mixm.fit(data['age'].reshape(-1,1))
combined = np.exp(mixm.score(x))

# Create Plots
fig, ax = plt.subplots(2,3, figsize=(13,8))
fig.subplots_adjust(wspace=0.3, hspace=0.3)

# Plots on first row
ax[0,0].hist(data[data.chd == 0]['age'].values, bins=25, color='orange')
ax[0,0].set_title('No CHD')

ax[0,1].hist(data[data.chd == 1]['age'].values, bins=25, color='lightblue')
ax[0,1].set_title('CHD')

ax[0,2].hist(data.age.values, bins=25, color='g')
ax[0,2].set_title('Combined')

for i in ax[0]:
    i.set_xlabel('Age')
    i.set_ylabel('Count')

# Plots on second row
ax[1,0].plot(x, no_chd, c='orange')

ax[1,1].plot(x, chd, c='lightblue')

ax[1,2].plot(x, no_chd, c='orange')
ax[1,2].plot(x, chd, c='lightblue')
ax[1,2].plot(x, combined , c='green')

for i in ax[1]:
    i.set_ylim(ymax = 0.10)
    i.set_xlabel('Age')
    i.set_ylabel('Mixture Estimate')

y = np.array([-0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53,
             0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22]).reshape(-1,1)
mixm = GMM(n_components=2, random_state=1, verbose=1)
mixm.fit(y)

mixm.means_

mixm.covars_

mixm.weights_

y_ = np.linspace(y.min(), y.max()).reshape(-1,1)
y_density = np.exp(mixm.score(y_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# left plot
ax1.hist(y, bins=16, normed=True, color='r')
ax1.set_ylim((0,1))
ax1.set_title('Normalized histogram')

# right plot
ax2.plot(y_, y_density, color='r', label='Gaussian densities')
ax2.scatter(y, mixm.score_samples(y)[1][:,0], c='g', label='Responsabilities\nComponent 1')
ax2.plot(y_, mixm.score_samples(y_)[1][:,0], '--g' )
sns.distplot(y, hist=False, kde=False, rug=True, ax=ax2)
ax2.set_ylim((-0.01,1.05))
ax2.set_title('Gaussian Mixture Model (2 components)')
ax2.legend()
ax2.text(mixm.means_[1],0.45 , '$\^\pi_1$ {}'.format(np.round(mixm.weights_[1], 3)))
ax2.text(mixm.means_[1],0.4 , '$\^\mu_1$ {}'.format(np.round(mixm.means_[1,0], 3)))
ax2.text(mixm.means_[1],0.35 , '$\^\sigma_1^2$ {}'.format(np.round(mixm.covars_[1,0], 3)))

ax2.text(mixm.means_[0],0.45 , '$\^\\pi_1$ {}'.format(np.round(mixm.weights_[0], 3)))
ax2.text(mixm.means_[0],0.4 , '$\^\mu_2$ {}'.format(np.round(mixm.means_[0,0], 3)))
ax2.text(mixm.means_[0],0.35 , '$\^\sigma_2^2$ {}'.format(np.round(mixm.covars_[0,0], 3)))

for i in fig.axes:
    i.set_xlabel('y')
    i.set_ylabel('density')

faithful = pd.read_csv('data/faithful.csv')
faithful.info()

mixm2 = GMM(n_components=2, covariance_type='full')
mixm2.fit(faithful)

x = np.linspace(1, 6)
y = np.linspace(40, 120)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T

# Z represents log probabilities of XX under mixm2 model
Z = -mixm2.score(XX).reshape(X.shape)
# Z2 represents the probability of observation belonging to component 1
Z2 = mixm2.predict_proba(XX)[:,0].reshape(X.shape)

# Contour plot of log likelihood of the model
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 1.2, 10), cmap=plt.cm.Reds)
# Contour plot where the component 1 probability = 0.5 (decision boundary)
CS2 = plt.contour(X, Y, Z2, levels=[0.5], colors='k')

for cplot in [CS, CS2]:
    plt.clabel(cplot, inline=True, fontsize=10, fmt='%1.1f')
    
plt.scatter(faithful.eruptions, faithful.waiting, label=None)
plt.legend([CS.collections[-1], CS2.collections[0]],
           ['Negative log-likelihood', 'Component decision boundery'])
plt.xlabel('Duration of eruption (minutes)')
plt.ylabel('Time to next eruption (minutes)')
plt.title("'Old Faithful' data set");


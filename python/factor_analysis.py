import numpy as np
import matplotlib.pyplot as plt
import utls 
import scipy.stats as ss
import seaborn as sns

sns.reset_orig()

utls.reset_plots()

get_ipython().run_line_magic('matplotlib', 'inline')

mu0 = 0
sigma0 = 1
z = np.random.normal(mu0,sigma0,5000) # latent variables
w = np.array([5,7]) # factor loading matrix (D x L)
mu = np.array([-1,1]) # centre
sigma = 3*np.eye(2) # this is constrained to be a diagonal matrix in general
x = np.array([np.random.multivariate_normal(zi*w+mu, sigma) for zi in z])

z_space = np.linspace(-3,3)

zi=2
p_zi = ss.norm.pdf(zi,mu0,sigma0)

X1_zi = zi*w[0] + mu[0] + np.linspace(-2,2)
X2_zi = zi*w[1] + mu[1] + np.linspace(-2,2)
X1_zi, X2_zi = np.meshgrid(X1_zi,X2_zi)
pos = np.empty(X1_zi.shape + (2,))
pos[:, :, 0] = X1_zi
pos[:, :, 1] = X2_zi
zi_density = utls.multivariate_gaussian(pos, mu=zi*w+mu, Sigma=sigma)

fig, axs = plt.subplots(1,2,figsize=(2*5,5))
axs = axs.ravel()

ax = axs[0]
ax.plot(z_space, ss.norm.pdf(z_space,mu0,sigma0),'-k')
ax.plot(zi, p_zi, 'gx')
ax.plot(np.ones(50)*zi, np.linspace(0,p_zi),'--g')
ax.set_xlabel('Latent variable, $z_i$')
ax.set_ylabel('Density, $p(z_i)$')
ax.set_ylim([0,0.42])

ax = axs[1]
k1 = sns.kdeplot(x[:,0],x[:,1],cmap="Reds",shade=True,shade_lowest=False,ax=ax)
ax.contour(X1_zi, X2_zi, zi_density, cmap='Greens')
ax.plot(w[0]*z_space+mu[0],w[1]*z_space+mu[1],'-k', label="$\mathbf{W}z_i+\mathbf{\mu}$")
ax.legend(prop={'size':20})
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_xlim([-3*w[0]+mu[0],3*w[0]+mu[0]])
ax.set_ylim([-3*w[1]+mu[1],3*w[1]+mu[1]])
plt.tight_layout()

import pandas as pd

d = pd.read_csv('../Data/04cars/04cars-fixed.csv')
d.head()

d = d.iloc[:,7:]

d.head()

d.shape

d_norm = (d - d.mean())/d.std(ddof = 1) # z-transform data

d_norm.reset_index(inplace=True)
d_norm.rename_axis({'index':'Name'}, axis="columns",inplace=True)

d_norm.head()

D = 11 # number of features
L = 2 # Dimension of the reduced-dimension space

from sklearn.decomposition import FactorAnalysis

f =FactorAnalysis(n_components=L)

X = d_norm.iloc[:,1:].as_matrix()

m = f.fit_transform(X) # fit model, and transform data, returning the means of the latent variables

m_scaled = m*0.1 # scale down the amplitudes of the scores, for plotting

feat_names = d_norm.columns[1:]

f.components_.shape

m.shape

sort_cars = m_scaled.argsort(axis=0)
max_x = sort_cars[:3]
min_x = sort_cars[-3:,:]
extreme_indicies = np.unique(np.ravel(np.vstack((max_x, min_x))))

cars_names = d_norm.Name

fig, ax = plt.subplots(1,1,figsize=(9,7.5))
for i in range(D):
    c1=f.components_[0,i]
    c2=-f.components_[1,i] # invert this axis (can always do this)
    ax.plot([0,c1],[0,c2],'-k')
    ax.annotate(feat_names[i],(c1,c2))
plt.plot(m_scaled[:,0],m_scaled[:,1],'.r')

for c in extreme_indicies:   
    ax.annotate(d_norm.loc[c].Name,m_scaled[c,:])           

ax.set_xlabel('Component 1, "price"')
ax.set_ylabel('Component 2, "fuel efficiency relative to size"')
ax.set_xlim([-0.6,1.2]);

f.components_

components_ordered = np.argsort(f.components_,axis=1)
components_ordered

features = d_norm.columns[1:]
features

features[components_ordered[0,-2:]]

features[components_ordered[1,-2:]]


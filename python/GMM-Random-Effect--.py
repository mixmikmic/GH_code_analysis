get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pymc3 as pm
import theano.tensor as T
import matplotlib
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sns.set_context("notebook", font_scale= 1.7) 

import warnings
warnings.filterwarnings("ignore")

sns.set(style="ticks")
from scipy import stats

# Data loading
df = pd.read_csv('df.csv')
df.head()

dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)] 

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12)
f, ax = plt.subplots(figsize=(7, 3))

ax.scatter(df.occupancy, df.Speed,c='#8B0000', alpha=0.12)
plt.xlim(0, 55)
plt.ylim(0, 85)

plt.ylabel("Speed $(mph)$", fontsize=13)
plt.xlabel("Occupancy (%)", fontsize=13)
sns.despine(offset=10)

bins=50

g = sns.FacetGrid(data = df, col='Ilane_n', col_wrap=2, aspect=1.5, sharex=True, sharey=True)
g.map(plt.hist, 'Speed', color='grey', normed=True, bins=bins, alpha = 0.25)
sns.despine(offset=10)
#plt.xlim(0,50)

plt.tight_layout()

g.axes[0].set_ylabel('Frequency')
g.axes[2].set_ylabel('Frequency')
g.axes[2].set_xlabel('Speed $(mph)$')
g.axes[3].set_xlabel('Speed $(mph)$')
#g.set_titles("lane {col_name}")
labels = ['Median', 'Inner-left', 'Inner-right', 'Shoulder']
for ax,j in zip(g.axes.flatten(),labels):
    ax.set_title(j)

g = sns.FacetGrid(data = df, col='Ilane_n', col_wrap=2, aspect=1.5, sharex=True, sharey=True)
g.map(plt.scatter,'occupancy', 'Speed',color='grey', alpha = 0.05)
sns.despine(offset=10)
plt.xlim(0,50)
plt.tight_layout()
g.axes[3].set_xlabel('Occupancy (%)')
g.axes[2].set_xlabel('Occupancy (%)')
g.axes[0].set_ylabel('Speed $(mph)$')
g.axes[2].set_ylabel('Speed $(mph)$')
labels = ['Median', 'Inner-left', 'Inner-right', 'Shoulder']
for ax,j in zip(g.axes.flatten(),labels):
    ax.set_title(j)

df_11 = df
df_11["lane_n"]  = df_11["Ilane_n"].astype('category')
from patsy import dmatrices
_, Z   = dmatrices('Speed ~ -1+lane_n', data=df_11, return_type='matrix')
Z      = np.asarray(Z) # mixed effect
nrandm = np.shape(Z)

n_lane = len(df.Ilane_n.unique())
Lane_idx = df.Ilane_n.values

n_lane

X = df.occupancy
Y = df.Speed

# Number of iteration
niter = 1000
k = 2                                                    # Number of components 
n =  X.shape[0]                                          # Number of Sample size
with pm.Model() as Mixture_regression:
    
    # Proportion in each mixture
    π = pm.Dirichlet('π', np.array([1]*k), testval=np.ones(k)/k)
    
    # Priors for unknown model parameters
    α = pm.Normal('α', mu=Y.mean(), sd=100,shape=k) #Intercept
    β = pm.Normal('β', mu=0, sd=100, shape=k) 
    σ  = pm.HalfCauchy('σ', 5,shape=k)  #Noise
    
    # Classifying each observation
    category = pm.Categorical('category', p=π, shape=n)
    
    σr = pm.HalfCauchy('σr', 5)
    μr = pm.Normal('μr', mu=0, sd = σr, shape=n_lane)
    
    # Expected value 
    #Choose beta and alpha based on category
    μ = α[category] + β[category]*X + T.dot(Z,μr)

    # Likelihood 
    Y_obs = pm.Normal('Y_obs', mu=μ, sd=σ[category], observed=Y)
    
    step1 = pm.NUTS([π, α, β, σ, σr, μr]) #
    step2 = pm.ElemwiseCategorical([category], values=range(k))
    trace = pm.sample(niter, [step1, step2], njobs=4, progressbar=True)

t1 = trace
pm.df_summary(t1,['π', 'α', 'β', 'σ', 'σr', 'μr']).round(4)

sns.set_context("notebook", font_scale= 1.0)
pm.traceplot(t1,['π', 'α', 'β', 'σ', 'σr', 'μr']);

# Creater posterior predictive samples
ppc = pm.sample_ppc(t1, model=Mixture_regression, samples=500)

sns.set_context("notebook", font_scale= 1.7) 

fig, axes = plt.subplots(1,1, figsize=(7,3))
plt.hist(Y,100, normed=True,alpha=0.5)
sns.kdeplot(Y, alpha=0.5, lw=1, c='g',linestyle='dashed')
sns.kdeplot(ppc['Y_obs'][1], c='k',label = 'Posterior predicted density')
#sns.distplot(Y, kde=False)
for i in range(100):
    sns.kdeplot(ppc['Y_obs'][i], alpha=0.1, c='k')
plt.xlabel('Speed (mph)'), plt.ylabel('Data density')
plt.xlim(10,80)
plt.legend(['Kernel density',
            'Posterior predicted density'
            ,'Observed traffic speed'],
           loc='best')
sns.despine(offset=10)






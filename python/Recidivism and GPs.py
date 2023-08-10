import numpy as np
import scipy as sp
import scipy.stats as sps

import pandas as pd
import GPy

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# load in data
inp = pd.read_excel("DATABASE Helmus 10 yr fixed HR-HN 09-29-2012.xlsx")
# give columns nicer names
inp.columns = ["score","rec"]
# drop the last row of totals
inp = inp.iloc[:-1]
# display a summary
inp.head()

# cross tabulate the data counting number of entries
ct = pd.crosstab(inp.score,inp.rec)
ct

# create our GP regression using an Exponential kernel
me = GPy.models.GPClassification(inp[["score"]].values, inp[["rec"]].values,
                                GPy.kern.Exponential(1))
# optimize hyperparameters
me.optimize()
# display summary of model
print(me)

# and again with a Matern covariance function
mm = GPy.models.GPClassification(inp[["score"]].values, inp[["rec"]].values,
                                GPy.kern.Matern52(1))
# optimize hyperparameters
mm.optimize()
# display summary of model
print(mm)

# pull in CDF function so we can display confidence intervals
from GPy.likelihoods.link_functions import std_norm_cdf

# get to posteriour of independent estimates with beta(1,1) prior
d = sps.beta(1+ct[1], 1+ct[0])

# generate some points on which to estimate GP
X = np.linspace(-2.5,11.5,101)

# get some better colors
pal = sns.color_palette()

# to make the plots look nice
with sns.axes_style("ticks"):
    fig,axs = plt.subplots(1,2,sharey=True,figsize=(14,5))

# loop over our two GPs
for i,(ax,m,kern) in enumerate(zip(axs,[mm,me],["Exp","Matern"])):
    # get a prediction from the GP, Y is the mean, Yv is the variance
    Y,Yv = m._raw_predict(X[:,None])
    Y,Yv = m._raw_predict(X[:,None])

    # draw the independent estimates with 95% CI
    ax.scatter(ct.index, d.mean(), label="Independent Beta(1,1)",
               facecolor='k', edgecolor=None)
    ax.vlines(ct.index, d.ppf(0.025), d.ppf(0.975), lw=1)

    # draw the GP mean and 95% CI
    ax.plot(X, std_norm_cdf(Y), color=pal[i], label="%s GP mean + 95%%CI"%(kern,))
    ax.fill_between(X,
                    std_norm_cdf(Y - 2*np.sqrt(Yv))[:,0],
                    std_norm_cdf(Y + 2*np.sqrt(Yv))[:,0],
                    alpha=0.2, color=pal[i], zorder=0)

    # add a figure legend
    ax.legend(loc="upper left")
    # set the axes up nicely
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(0,1)
    ax.set_xlabel("score")

# just a single label on the y axis
axs[0].set_ylabel("recidivism probability")

# remove lines on the top and right of plot and squeeze everything up nicely
sns.despine()
plt.tight_layout()


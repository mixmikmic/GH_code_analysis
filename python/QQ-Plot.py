import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pylab
import scipy.stats as stats
pylab.rcParams['figure.figsize'] = (5,5)
np.random.seed(3212967995)

data = np.random.normal(0,1, 50)
sm.qqplot(data,stats.norm,line="r");

pylab.rcParams['figure.figsize'] = (10,10)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
sm.qqplot(np.random.normal(0,1,10),stats.norm,line="r", ax=ax1);
sm.qqplot(np.random.normal(0,1,25),stats.norm,line="r", ax=ax2);
sm.qqplot(np.random.normal(0,1,100),stats.norm,line="r", ax=ax3);
sm.qqplot(np.random.normal(0,1,1000),stats.norm,line="r", ax=ax4);

def drawQQForNormal(mu, sigma, diagramPosition,ax1=None):
    if ax1==None: 
        ax1 = plt.subplot(diagramPosition)
    else:
        ax1 = plt.subplot(diagramPosition,sharex=ax1)
    sm.qqplot(np.random.normal(mu,sigma,100),stats.norm,line="r", ax=ax1);
    plt.grid(True)
    plt.axis((-4,4,-4,4))
    plt.setp(ax1.get_xticklabels(), fontsize=12)
    return ax1
    
ax1 = drawQQForNormal(0,1,221)
drawQQForNormal(2,1,222,ax1)
drawQQForNormal(0,0.5,223,ax1)
drawQQForNormal(2,0.5,224,ax1)

pylab.rcParams['figure.figsize'] = (5,5)
data = np.random.standard_t(df=2,size=50)
qqPlot = sm.qqplot(data,stats.norm,line="r")

data_1 = np.random.normal(0,1, 25)
data_2 = np.random.normal(3,0.5, 25)
data = np.concatenate((data_1, data_2), axis=0)
qqPlot = sm.qqplot(data,stats.norm,line="r")
qqPlot.show()


from pymc3 import Model, Normal, Uniform, NUTS, sample, find_MAP, traceplot, summary, df_summary, trace_to_dataframe
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x = np.array([1,2,3,4,5,6,7,8,9,10])
y =np.array([5.19, 6.56, 9.19, 8.09, 7.6, 7.08, 6.74, 9.3, 9.98, 11.5])
df = pd.DataFrame({'x':x, 'y':y})

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')

basic_model = Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=100)
    beta = Normal('beta', mu=0, sd=100)
    sigma = Uniform('sigma', lower=0, upper=10000)

    # Expected value of outcome
    mu = alpha + beta*x 

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y)

#find_MAP can be used to establish a useful starting point for the sampling
map_estimate = find_MAP(model=basic_model)

print(map_estimate)

import scipy

with basic_model:

    # obtain starting values via MAP
    start = find_MAP(fmin=scipy.optimize.fmin_powell)

    # draw 2000 posterior samples
    trace = sample(2000, start=start) 

traceplot(trace, lines={k: v['mean'] for k, v in df_summary(trace).iterrows()})

summary(trace)

df_summary(trace, alpha =0.1)

def plot_posterior_cr(mdl, trc, rawdata, xlims, npoints=1000):  
    '''
    Convenience fn: plot the posterior predictions from mdl given trcs
    '''

    ## extract traces
    trc_mu = trace_to_dataframe(trc)[['alpha','beta']]
    trc_sd = trace_to_dataframe(trc)['sigma']

    
    ## recreate the likelihood
    x = np.linspace(xlims[0], xlims[1], npoints).reshape((npoints,1))
    X = x ** np.ones((npoints,2)) * np.array([0, 1])
    like_mu = np.dot(X,trc_mu.T) + trc_mu.median()['alpha']
    like_sd = np.tile(trc_sd.T,(npoints,1))
    like = np.random.normal(like_mu, like_sd)

    ## Calculate credible regions and plot over the datapoints
    dfp = pd.DataFrame(np.percentile(like,[2.5, 25, 50, 75, 97.5], axis=1).T
                         ,columns=['025','250','500','750','975'])
    dfp['x'] = x

    pal = sns.color_palette('Purples')
    f, ax1d = plt.subplots(1,1, figsize=(7,7))
    ax1d.fill_between(dfp['x'], dfp['025'], dfp['975'], alpha=0.5
                      ,color=pal[1], label='CR 95%')
    ax1d.fill_between(dfp['x'], dfp['250'], dfp['750'], alpha=0.4
                      ,color=pal[4], label='CR 50%')
    ax1d.plot(dfp['x'], dfp['500'], alpha=0.5, color=pal[5], label='Median')
    _ = plt.legend()
    _ = ax1d.set_xlim(xlims)
    _ = sns.regplot(x='x', y='y', data=rawdata, fit_reg=False
            ,scatter_kws={'alpha':0.8,'s':80, 'lw':2,'edgecolor':'w'}, ax=ax1d)

xlims = (df['x'].min() - np.ptp(df['x'])/10  
                 ,df['x'].max() + np.ptp(df['x'])/10)

plot_posterior_cr(basic_model, trace, df, xlims)


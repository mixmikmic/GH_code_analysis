import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import summary_table
get_ipython().magic('matplotlib inline')

from matplotlib import rc
rc('font', **{'family':'sans-serif',
    'sans-serif':['Helvetica'],
    'monospace': ['Inconsolata'],
    'serif': ['Helvetica']})
rc('text', **{'usetex': True})
rc('text', **{'latex.preamble': '\usepackage{sfmath}'})

year_distance = pd.read_csv("../data/year_distance.csv")
year_distance

# We don't need to add a slope constant to the design matrix (A) if we're using patsy-style formulas.
fit = sm.ols(
     formula='Distance ~ Year',
     data=year_distance).fit()
fit.summary()

fit.params

fit.resid

# OLS confidence intervals
st, data, ss2 = summary_table(fit, alpha=0.05)

fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, axisbg='none')
# this will save as 728px (max width for a Github README) when dpi is set to 100
fig.set_size_inches(7.28, 5.25)

# Original measured values
orig = plt.scatter(
    year_distance['Year'],
    year_distance['Distance'],
    color='#006600',
    edgecolor='#333333',
    marker='o',
    s=80,
    lw=1,
    zorder=2,
    alpha=1.)
# we need the comma here: http://stackoverflow.com/questions/11983024/matplotlib-legends-not-working
# Line of best fit
fitted, = plt.plot(
    year_distance['Year'],
    fit.predict(),
    color='#CC00CC',
    zorder=1)
ax.set_xlabel('Year')
ax.set_ylabel('Measured Distance')
# let's predict some data
years = [1995, 1998.5, 2001.5, 2004.5]
predicted = fit.predict(
    pd.DataFrame({
        'Intercept': np.ones(len(years)),
        'Year': years}))
pred = plt.scatter(
    years,
    predicted,
    color='#33CCFF',
    edgecolor='#333333',
    marker='o',
    s=80,
    lw=1,
    alpha=1.0,
    zorder=2)

cil, = plt.plot(year_distance['Year'], predict_ci_low, 'r--', lw=1, alpha=0.5)
ciu, = plt.plot(year_distance['Year'], predict_ci_upp, 'r--', lw=1, alpha=0.5)
# mcil, = plt.plot(year_distance['Year'], predict_mean_ci_low, 'b--', lw=1, alpha=0.5)
# mciu, = plt.plot(year_distance['Year'], predict_mean_ci_upp, 'b--', lw=1, alpha=0.5)

ax.fill_between(year_distance['Year'], predict_ci_low, predict_ci_upp, alpha=0.3)

leg = plt.legend(
    (orig, fitted, pred, ciu),
    ('Measurements', r'Best fit ($\sigma$:~%0.4f)' % fit.mse_resid, 'Predicted', 'Conf. Interval (95\%)'),
    loc='upper left',
    scatterpoints=1,
    fontsize=9)

plt.title('Distance Measured Each Year')
leg.get_frame().set_alpha(0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(b=None)
plt.tight_layout()
plt.savefig('../OLS.png', format="png", bbox_inches='tight', alpha=True, transparent=True, dpi=100)




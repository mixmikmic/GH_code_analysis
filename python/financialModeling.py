# Load the libraries.

get_ipython().magic('matplotlib inline')
import warnings
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt 
from numpy import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
#import Quandl
from IPython.display import Image
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sma
import patsy
from statsmodels.graphics.api import abline_plot
import numpy.linalg as linalg
import pymc3 as pm
from mpl_toolkits.mplot3d import Axes3D
warnings.simplefilter('ignore')
sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, rc=None)

# When I teach, I stand in front of a sea of these.

Image(url = 'https://www.truthinadvertising.org/wp-content/uploads/2015/01/WEB-Apple-logo--620x350.jpg')

# Grab 10 years of Apple and NASDAQ data using pandas.io.data from Yahoo!Finance.
# Plot
# Note Apple's dip during the financial crisis of '07 - '09 as well as its steep decline in '13.
# NASDAQ dips during the financial crisis but has a tremendous run starting in early '09.

start, end = dt.datetime(2006, 1, 1), dt.datetime(2016, 12, 31)
aapl_all = web.DataReader('aapl', 'yahoo', start, end)
nasdaq_all = web.DataReader('^ixic', 'yahoo', start, end)
aapl = aapl_all['Adj Close']
nasdaq = nasdaq_all['Adj Close']

aapl.plot()
plt.title('AAPL ($/Share)', fontsize = 20)

nasdaq.plot()
plt.title('NASDAQ', fontsize = 20)

# Calculate log returns.
# Display mean and volatility.
# Note the differences: AAPL has on average higher daily returns and more volatility.

aapl_returns = ((np.log(aapl / aapl.shift(1))).dropna())
nasdaq_returns = (np.log(nasdaq / nasdaq.shift(1))).dropna()

print (aapl_returns.mean(), aapl_returns.std())
print (nasdaq_returns.mean(), nasdaq_returns.std())

# Display histogram of AAPL returns.
# Do they look normally distributed?

plt.figure(figsize=(8,8))
plt.hist(aapl_returns, bins=150, normed=True, color='blue')
plt.title('Histogram of AAPL Daily Returns Since 2006', fontsize=20)
plt.ylabel('%', fontsize=8)
plt.axvline(0, color='red')
plt.xlim(-0.2, 0.2)
plt.ylim(0, 60)

plt.figure(figsize=(8,8))
plt.hist(nasdaq_returns, bins=150, normed=True, color='blue')
plt.title('Histogram of NASDAQ Daily Returns Since 2006', fontsize=20)
plt.ylabel('%', fontsize=8)
plt.axvline(0, color='red')
plt.xlim(-0.2, 0.2)
plt.ylim(0, 60)

aapl_returns = pd.DataFrame(aapl_returns)
nasdaq_returns = pd.DataFrame(nasdaq_returns)

# Generate a scatterplot.

plt.figure(figsize = (8,8))
plt.scatter(aapl_returns, nasdaq_returns)
plt.title('CAPM Data', fontsize = 20)
plt.xlabel('Log Returns of NASDAQ', fontsize = 10)
plt.ylabel('Log Returns of AAPL', fontsize = 10)
plt.xlim([-0.1, 0.1])
plt.ylim([-0.1, 0.1])

# Merge and create DF for regression.

aapl_r = pd.DataFrame(aapl_returns)
nasdaq_r = pd.DataFrame(nasdaq_returns)
data = pd.merge(nasdaq_r, aapl_r, left_index=True, right_index=True)
data.head()
data.rename(columns={'Adj Close_x':'nasdaq', 'Adj Close_y':'aapl'}, inplace=True)
mod = smf.ols(formula='aapl ~ nasdaq', data = data).fit()
print(mod.summary())

# CAPM data with best linear fit.

figure, ax = plt.subplots(figsize=(8,8))
ax.scatter(aapl_returns, nasdaq_returns)
mod = smf.ols(formula='aapl ~ nasdaq', data = data).fit()
abline_plot(model_results=mod, ax=ax, color='red')

ax.set_title('CAPM Data', fontsize = 20)
ax.set_ylabel('Log Returns of AAPL', fontsize = 10)
ax.set_xlabel('Log Returns of NASDAQ', fontsize = 10)
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])

# Hypothesis testing.

print(mod.f_test("Intercept = 0"))
print(mod.f_test("nasdaq = 1"))
print(mod.f_test("nasdaq = 1, Intercept = 0"))

# A Bayesian approach to decision making might be fruitful.

with pm.Model() as model:
    # alpha, beta, and sigma are the hyperparameters over which we have our priors, in this case they are flat priors.    
    alpha = pm.Normal('alpha', mu=0, sd=20)
    beta = pm.Normal('beta', mu=0, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    
    # y_est is the specification of the Bayesian model to be estimated.  It is simply our CAPM.
    y_est = alpha + beta * nasdaq_returns
    
    # likelihood is the likelihood function, here it is normal to be used with conjugate priors.    
    likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=aapl_returns)
    
    # We use the Maximum a Posteriori (MAP) values as starting values for the MCMC sampling.
    start = pm.find_MAP()
    step = pm.NUTS(state=start)
    trace = pm.sample(1000, step, start=start, progressbar=True)

# Show results after burn in of 200 MCMC replications.

fig = pm.traceplot(trace[199:1000], lines={'alpha': 0, 'beta': 1})
plt.figure(figsize = (10, 10))

# Having simulated the entire posterior probability distributions, 
# we can calculate useful metrics.

# Start with averages and 95% credible intervals.
# Note their simularity to the least squares results.

print("Average alpha and its 95% credible interval are:", np.mean(trace['alpha'][199:1000]), np.percentile(trace['alpha'][199:1000], (2.5, 97.5)))
print("Average beta and its 95% credible interval are:", np.mean(trace['beta'][199:1000]), np.percentile(trace['beta'][199:1000], (2.5, 97.5)))
print()
print(mod.summary())

# Having simulated the entire posterior probability distributions, we can calculate useful metrics.

print ("The probability that alpha is greater than zero is", np.mean(trace['alpha'][199:1000] > 0.0))
print ("The probability that beta is less than one is", np.mean(trace['beta'][199:1000] < 1.0))
print("The joint probability is", np.all([[trace['alpha'][199:1000] > 0.0], [trace['beta'][199:1000] < 1.0]], axis = 0).mean())

# Suppose one wanted to model 10-year US Treasurys.  (Yes, the spelling is correct.)  Start with observed rates.

start, end = dt.datetime(2006, 1, 1), dt.datetime(2016, 12, 31)
rates = web.DataReader('^TNX', 'yahoo', start, end)

rates['Adj Close'] = rates['Adj Close'] / 100
rates['Adj Close'].plot()
plt.title('10 Years of 10-Year U.S. Treasurys', fontsize=20)

# Calculate average and standard deviation (as a measure of volatility).

mean = rates['Adj Close'].mean() 
vol = rates['Adj Close'].std() 
print(rates['Adj Close'].mean(), rates['Adj Close'].std())

def v(r0, kappa, theta, sigma, Time, N, randomseed):
    np.random.seed(randomseed)
    dt = float(Time / N)
    rates = [r0]
    for i in range(N):
        dr = kappa * (theta - rates[-1]) * dt + sigma * np.random.normal()
        rates.append(rates[-1] + dr) 
    return range(N+1), rates

# Use Vasicek model with observed data.  
# We see there's an immediate problem: negative nominal interest ratees.
# This used to be quaint binding constraint.

x, y = v(mean, 0.30, mean, vol, 10., 500, 2272007)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.axhline(0, color = 'red')
plt.title('A Simulated Path of Nominal 10-Year Treasuries', fontsize=20)
plt.show()

def cir(r0, kappa, theta, sigma, Time, N, randomseed):
    np.random.seed(randomseed)
    dt = float(Time / N)
    rates = [r0]
    for i in range(N):
        dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(rates[-1]) * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

# This is graphed using the same y-range as the actual 10-year rates in the graph above.  
# Note that the model is able to capture long-lived swings, as well as to meet the ZLB condition.

x, y = cir(mean, 0.30, mean, vol, 10., 500, 672007)

plt.plot(x,y)
#plt.axhline(0, color = 'red')
plt.title('A Simulated Path of Nominal 10-Year Treasuries', fontsize=20)
plt.show()

def d(r0, alpha, sigma, Time, N, randomseed):
    np.random.seed(randomseed)
    dt = float (Time / N)
    rates = [r0]
    for i in range(N):
        dr = alpha * rates[-1] * dt + sigma * rates[-1] * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

x, y = d(mean, 0.001, vol, 10., 500, 972008)

plt.plot(x,y)
#plt.axhline(0, color = 'red')
plt.title('A Simulated Path of Nominal 10-Year Treasuries', fontsize=20)
plt.show()

def rb(r0, theta, sigma, Time, N, randomseed):
    np.random.seed(randomseed)
    dt = float(Time / N)
    rates = [r0]
    for i in range(N):
        dr = theta * rates[-1] * dt + sigma * rates[-1] * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

# The return of Volcker.

x, y = rb(mean, 0.01, vol, 10., 500, 1066)

plt.plot(x,y)
#plt.axhline(0, color = 'red')
plt.title('A Simulated Path of Nominal 10-Year Treasuries', fontsize=20)
plt.show()

def bs(r0, kappa, theta, sigma, Time, N, randomseed):
    np.random.seed(randomseed)
    dt = float (Time / N)
    rates = [r0]
    for i in range(N):
        dr = kappa * (theta - rates[-1]) * dt + sigma * rates[-1] * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

x, y = bs(mean, 0.30, mean, vol, 10., 500, 10241929)

plt.plot(x,y)
#plt.axhline(0, color = 'red')
plt.title('A Simulated Path of Nominal 10-Year Treasuries', fontsize=20)
plt.show()

def ev(r0, eta, alpha, sigma, Time, N, randomseed):
    np.random.seed(randomseed)
    dt = float (Time / N)
    rates = [r0]
    for i in range(N):
        dr = rates[-1] * (eta - alpha * np.log(rates[-1])) * dt + sigma * rates[-1] * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

x, y = bs(mean, 0.350, mean, vol, 10., 500, 9152008)

plt.plot(x,y)
#plt.axhline(0, color = 'red')
plt.title('A Simulated Path of Nominal 10-Year Treasuries', fontsize=20)
plt.show()


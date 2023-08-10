# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.
get_ipython().magic('matplotlib inline')
# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import requests
import json
import datetime

pjmdf = pd.read_csv('rawdf_pjm.csv')
pjmdf.head()

pjmdf.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)
pjmdf.head()

pjmdf.shape

dfpn7601 = pjmdf[pjmdf['pnodeId']==32417601]
df = dfpn7601[:1460]
df = df.drop('pnodeId',1)
df = df.drop('publishDate',1)
df = df.fillna(0)
lmp = df.stack()
rng = pd.date_range('1/1/2008', periods = 35040, freq = 'H')
lmp.index = rng

lmp.plot()
plt.ylabel('$/MWh')

pjm_load = pd.read_csv('data/PJM_hourly_load_08_15.csv')
ldf = pjm_load[:1460]
ldf = ldf.drop('DATE',1)
ldf = ldf.drop('COMP',1)
ldf = ldf.drop('Unnamed: 26',1)
ldf = ldf.drop('Unnamed: 27',1)
ldf = ldf.fillna(0)
load = ldf.stack()
rng = pd.date_range('1/1/2008', periods = 35040, freq = 'H')
load.index = rng

plt.scatter(load,lmp)
plt.xlabel('MWh of demand')
plt.ylabel('$/MWh')

plt.scatter(np.log(load),np.log(lmp))
plt.xlabel('log(MWh of demand)')
plt.ylabel('$/MWh')

ldf = pd.DataFrame(load)
ldf.hist(bins=100)
plt.title('Frequency of total demand')
plt.xlabel('MWh of demand')
plt.ylabel('# of hours')

lmpdf = pd.DataFrame(lmp)
lmpdf.hist(bins=100)
plt.title('Frequency of price')
plt.xlabel('$/MWh')
plt.ylabel('# of hours')

pjm_wind = pd.read_csv('data/PJM_Wind/PJM_hourly_wind_08_11.csv')
wdf = pjm_wind[:1460]
wdf = wdf.drop('DATE',1)
wdf = wdf.drop('COMP',1)
wdf = wdf.fillna(0)
wind = wdf.stack()
rng = pd.date_range('1/1/2008', periods = 35040, freq = 'H')
wind.index = rng

plt.scatter(wind, lmp)

plt.scatter(wind, np.log(lmp))

winddf = pd.DataFrame(wind)
winddf.hist(bins=100)
plt.title('Frequency of wind gen')
plt.xlabel('MWh')
plt.ylabel('# of hours')

lmpdf = pd.DataFrame(lmp)
loaddf = pd.DataFrame(load)
winddf = pd.DataFrame(wind)

#lmpdf.columns = ['date','lmp']

lmpdf['wind'] = winddf
lmpdf['load'] = loaddf
lmpdf.columns = ['lmp','wind', 'load']

#lmpdf.to_csv('../df.csv')
#dfNG.to_csv('../ngdf.csv')

import statsmodels.formula.api as smf

lm = smf.ols(formula = 'lmp ~ load + wind', data = lmpdf).fit()
lm.summary()

df = pd.read_csv('data/LMP_features_08_12.csv')
df = df.set_index(['date'])

lm = smf.ols(formula = 'lmp ~ load + ng + wind', data = df).fit()

lm.summary()

dflog = df
dflog['log_lmp'] = np.log1p(df['lmp'])
dflog['log_wind'] = np.log1p(df['wind'])
dflog['log_load'] = np.log1p(df['load'])
dflog['log_ng'] = np.log1p(df['ng'])
dflog = dflog.fillna(0)

dflog = dflog[df['lmp']>10]
y = dflog['log_lmp']
x = dflog['log_ng']
#y = dflog['lmp']
#x = dflog['ng']

plt.scatter(x,y)

y = dflog['lmp']
x = dflog['load']

plt.scatter(x,y)

lm = smf.ols(formula = 'lmp ~ ng + load', data = dflog).fit()

lm.summary()

df['month'] = pd.to_datetime(df.index).month

df.set_index(['month',df.index])
#df.head()
df.head()


#model = PanelOLS(y=df['lmp'], x=df['ng'], time_effects=True)
#df.ix[1]
model = smf.MixedLM.from_formula("lmp ~ ng", df, groups=df["month"])
result = model.fit()

print result.summary()

from pandas.stats.plm import PanelOLS
#model = PanelOLS(y=p['lmp'], x=p['ng'], time_effects=True)
#m=pd.ols(y=p['lmp'],x={'ng':p['ng'],'ld':p['load'],'wd':p['wind']},time_effects=True)
m=PanelOLS(y=p['lmp'],x={'ng':p['ng'],'ld':p['load'],'wd':p['wind']},time_effects=True)
m

df.head()

lm = smf.ols(formula = 'lmp ~ ng + load + wind', data = dflog).fit()
lm.summary()

lm = smf.ols(formula = 'log_lmp ~ log_load + log_ng + wind', data = dflog).fit()
lm.summary()
#dflog

# read in dataframe from csv, set index and add month column for fixed effects
df = pd.read_csv('fulldf.csv')
df.set_index(['Date'],inplace = True)
df['hour'] = pd.to_datetime(df.index).hour
df['month'] = pd.to_datetime(df.index).month
df['dow'] = pd.to_datetime(df.index).dayofweek
df.set_index(['dow',df.index],inplace=True)
df.head()

# Outlier removal
print df.shape
df = df[df['lmp']>0]
df = df[df['lmp']<200]
df = df[df['load']>0]
print df.shape

# OLS regression
lm = smf.ols(formula = 'lmp ~ load + ng + wind + coal', data = df).fit()
lm.rsquared

# Panel regression
from pandas.stats.plm import PanelOLS
p = df.to_panel()
m=PanelOLS(y=p['lmp'],x={'ng':p['ng'],'ld':p['load'],'wd':p['wind'], 'cl':p['coal']},time_effects=True, intercept=True)
m

df = df[df['lmp']>0]
df = df[df['lmp']<100]
df['lmp'].hist(bins=100)
plt.title('Frequency of LMP')
plt.xlabel('$/MWh')
plt.ylabel('# of hours')

lm = smf.ols(formula = 'lmp ~ load + ng + wind + coal', data = df).fit()
lm.summary()

dflog = df
dflog['log_lmp'] = np.log1p(df['lmp'])
dflog['log_wind'] = np.log1p(df['wind'])
dflog['log_load'] = np.log1p(df['load'])
dflog['log_ng'] = np.log1p(df['ng'])
dflog['log_coal'] = np.log1p(df['coal'])
dflog = dflog.fillna(0)
dflog = dflog[dflog['lmp']>0]
dflog.head()

lm = smf.ols(formula = 'lmp ~ load + ng + wind + coal', data = df).fit()
lm.summary()




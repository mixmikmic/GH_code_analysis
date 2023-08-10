import snsims

import sncosmo

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

data = sncosmo.load_example_data()

model = sncosmo.Model(source='salt2')

model.set(**data.meta)

print model

fig = sncosmo.plot_lc(data, model=model, color='k')

lc = snsims.LightCurve(data.to_pandas())

lc.lightCurve

newdata = data.to_pandas()

newdata.columns

newdata.rename(columns={'time': 'expMJD'}, inplace=True)

newdata

newlc = snsims.LightCurve(newdata)

newlc.columnAliases

newlc._lightCurve.columns

snsims.aliasDictionary(newlc._lightCurve.columns, newlc.columnAliases)

snsims.aliasDictionary(data.columns, newlc.columnAliases)

newlc.lightCurve

sncosmo.plot_lc(newlc.snCosmoLC(), model, color='k')

lc.coaddedLC(coaddTimes=10.)

lcc = lc.lightCurve.copy()

grouped = lcc.groupby(['band'])

import numpy as np

grouped.agg({'mjd':np.mean, 'band': 'count'})

sncosmo.plot_lc(lc.snCosmoLC(coaddTimes=10., mjdBefore=1.), model, color='k')

lc.lightCurve.query('band == "sdssg"')

all(lc.coaddedLC(coaddTimes=None) == lc.lightCurve)

lc.coaddedLC(coaddTimes=20., mjdBefore=1)

lc.coaddedLC(coaddTimes=20., mjdBefore=1).reset_index('band', inplace=True)

12* 256 * 256 /2 * 5/ 1000

12* 256 * 256 /2  / 1e6

8 * 200 * 5

9216 /60. / 218.

import analyzeSN as ans

import sncosmo

sncosmo.get_bandpass('lsst_g')




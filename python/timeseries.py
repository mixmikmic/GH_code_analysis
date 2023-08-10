import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
a = pd.read_csv('time series sample.csv')
a.head()

a['Open'].plot(x=a['Date'].astype(str))
plt.show()

import split as sp
k = sp.rollingmean(a,'Open',3)
k.plot(x=a['Date'].astype(str))
plt.show()

k = sp.rollingstd(a,'Open',3)
k.plot(x=a['Date'].astype(str))
plt.show()

import muloutlier as ml 
ml.timeoutlier(a,'High',7)

import timeseries as ts
ts.decompose(a,'Open',10)

ts.check(a,'Open')

ts.lmtestcheck(a,'Open',40)

ts.checkdb(a,'Open')

ts.freq(a,'Open',a['Open'].count())


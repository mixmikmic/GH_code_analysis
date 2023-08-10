get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

df=pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/austres.csv ")

df=df.drop('Unnamed: 0',1)

df.head()

df.tail()

len(df)

df.shape[0]

start = datetime.datetime.strptime("1971-03-31", "%Y-%m-%d")
print(start)

#!pip install arrow

start = datetime.datetime.strptime("1971-03-31", "%Y-%m-%d")
print(start)

date_list = [start + relativedelta(months=x) for x in range(0,3*df.shape[0])]

date_list

len(date_list)

c2=[]
for i in range(0,3*len(df),3):
    c2.append(date_list[i])

c2

len(c2)

df['index'] =c2
df.set_index(['index'], inplace=True)
df.index.name=None

df.head()

df=df.drop('time',1)

df.tail()

df.austres.plot(figsize=(12,8), title= 'Number of Australian Residents', fontsize=14)
plt.savefig('austrailian_residents.png', bbox_inches='tight')

decomposition = seasonal_decompose(df.austres, freq=4)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
print(p)

import itertools
import warnings

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
print(pdq)

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

y=df

warnings.filterwarnings("ignore") # specify to ignore warning messages
c4=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            c4.append('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

warnings.filterwarnings("ignore") # specify to ignore warning messages
c3=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            c3.append( results.aic)
        except:
            continue

c3

len(c3)

import numpy as np
index_min = np.argmin(c3)

index_min

c4[index_min]

type(c4[index_min])

from statsmodels.tsa.x13 import x13_arima_select_order

order1=c4[index_min][6:13]
order1

type(order1)

order1=[int(s) for s in order1.split(',')]
order1

type(order1)

seasonal_order1=c4[index_min][16:27]
seasonal_order1

seasonal_order1=[int(s) for s in seasonal_order1.split(',')]
seasonal_order1

mod = sm.tsa.statespace.SARIMAX(df.austres, trend='n', order=order1, seasonal_order=seasonal_order1)

results = mod.fit()
print (results.summary())

results.predict(start=78,end=99)

results.predict(start=78,end=99).plot()




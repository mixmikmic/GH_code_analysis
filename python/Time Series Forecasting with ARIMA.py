import pandas as pd
import warnings
import itertools
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

data = pd.read_csv('bentley_no_negative.csv')
data['Date'] = pd.to_datetime(pd.Series(data['Stock Trans Date']), format="%d/%m/%Y")
data = data.drop(['Stock Trans Date'],1)

best_sellers = ['BBQ.COVER/3',
                'LAD.06',
                'LAD.07',
                'PAT.01/C',
                'CAN.01/C',
                'GL/NB.01',
                'GL/BB.04',
                'CAIR.02/SS',
                'GL/SKB.01',
                'GL/FBG.02']

inputs = 'LAD.06'

#Select by partial string
#inputs = best_sellers[0]
select_data = data[data['Stock Product Code'].str.contains(inputs)]

select_data.head()

select_data = select_data.set_index('Date')

select_data.head()

# convert object into int datatype
select_data['Stock Qty Sales All']=select_data['Stock Qty Sales All'].astype(int)

#y = select_data['Stock Qty Sales All'].resample('MS').mean()
y = select_data['Stock Qty Sales All'].resample('W').sum()

y.isnull().sum()

y = y.fillna(0)

y.isnull().sum()

y.to_csv(inputs+'.csv')

y.plot(figsize=(15,6))
plt.title(inputs)
plt.show()

import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 11,9

decomposition = sm.tsa.seasonal_decompose(y,model='additive')
fig = decomposition.plot()
#plt.savefig("/Users/cozg3/Documents/Time-Series-Forecasting/decom.png")
plt.show()

p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

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
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['2012':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Stock Qty Sales All')
plt.legend()
plt.title(inputs)
plt.savefig('1.png')
plt.show()

data = pred.predicted_mean

#data.to_csv('pred.csv')

y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Get forecast 20 steps ahead in future
pred_uc = results.get_forecast(steps=20)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Stock Qty Sales')

plt.legend()
plt.show()

data = pred_uc.predicted_mean

data.columns = ["Date","Prediction"]

print(data)

data = pd.DataFrame([data], columns = ["A","B"])

data.head()

product_code = 'BBQ'
data['Name'] = product_code

name = 'Orange'

data.to_csv(name+'.csv')




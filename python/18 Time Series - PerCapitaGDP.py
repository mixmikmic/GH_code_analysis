import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
pd.set_option('precision', 4)

plt.rcParams["figure.figsize"] = (15, 6)

get_ipython().magic('matplotlib inline')

gdp= pd.read_csv("/data/gdp.csv")
gdp.info()

gdp.head()

gdp = gdp.set_index(pd.to_datetime(gdp.Year, format="%Y"))
gdp.head()

gdp.iso2c.unique()

plt.figure(figsize = (15, 8))
gdp.groupby(["Country"]).GDP.plot(legend = True);

usa = gdp.query("iso2c == 'US'").PerCapGDP
usa.head()

usa.plot()

usa_rolling_mean = usa.rolling(window = 10, center = False).mean() 
usa_rolling_std = usa.rolling(window = 10, center = False).std() 

plt.plot(usa, label = "actual")
plt.plot(usa_rolling_mean, label = "moving avg")
plt.plot(usa_rolling_std, label = "Std")

plt.legend()

usa_log = np.log(usa)
usa_rolling_mean = usa_log.rolling(window = 10, center = False).mean() 
usa_rolling_std = usa_log.rolling(window = 10, center = False).std() 

plt.plot(usa_log)
plt.plot(usa_rolling_mean)
plt.plot(usa_rolling_std)

from statsmodels.tsa.stattools import adfuller, acf, pacf

def test_stationary(tseries):
    dftest = adfuller(tseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
            index=['Test Statistic','p-value'
                   ,'#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationary(usa)

test_stationary(usa_log)

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(usa)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(usa, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

usa_diff = usa - usa.shift(periods=1)
usa_diff.dropna(inplace=True)
plt.plot(usa_diff)
test_stationary(usa_diff)

#ACF and PACF plots

plt.figure(figsize=(15, 6))
def plot_acf_pacf(ts):
    lag_acf = acf(ts, nlags=10)
    lag_pacf = pacf(ts, nlags=10, method='ols')

    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-7.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=7.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-7.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=7.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    
usa_diff = usa - usa.shift(periods=1)
usa_diff.dropna(inplace=True)
plot_acf_pacf(usa_diff)

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(usa, order=(2, 1, 0))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(usa_diff, label = "Original")
plt.plot(results_ARIMA.fittedvalues, color='red', label = "fitted")
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - usa_diff)**2))
plt.legend()

print(results_ARIMA.summary())

residuals = pd.DataFrame(results_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA = pd.Series(usa.ix[0], index=usa.index)
predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum, fill_value=0)

plt.plot(usa, label = "Actual")
plt.plot(predictions_ARIMA, label = "ARIMA fitted")
plt.legend()

from sklearn.metrics import mean_squared_error

size = int(len(usa) - 5)
train, test = usa[0:size], usa[size:len(usa)]
history = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)

predictions_series = pd.Series(predictions, index = test.index)

#rolling one-step out-of-sample
fig, ax = plt.subplots()
ax.set(title='GDP Per Capita forecasting', xlabel='Date', ylabel='Euro into USD')
ax.plot(usa, 'o', label='observed')
ax.plot(predictions_series, 'g', label='forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')




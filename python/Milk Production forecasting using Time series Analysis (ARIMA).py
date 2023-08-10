import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Step 1 - First we will clean the data

df = pd.read_csv('monthly-milk-production-pounds-p.csv')

df.head()

df.columns  = ['Month','Milk in pounds for Cow']

df.tail()

df.drop(168,axis=0,inplace=True)

df.tail()

df['Month'] = pd.to_datetime(df['Month'])

df.head()

df.set_index('Month',inplace=True)

df.head()

df.index

df.describe().transpose()

#Step2 - We will visualize the data now

df.plot()

time_series = df['Milk in pounds for Cow']

type(time_series)

time_series.rolling(12).mean().plot(label='12 Month Rolling Mean')
time_series.rolling(12).std().plot(label='12 Month Rolling Std')
time_series.plot()
plt.legend()

# Now let us check the ETS decomposition plot (Trend, Seasonallity and Residual)

from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(time_series)

fig = decomp.plot()
fig.set_size_inches(15,12)

from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Milk in pounds for Cow'])

def adf_check(time_series):
    
    result = adfuller(time_series)
    print("Augmented Dicky-Fuller Test")
    labels = ['ADF Test Statistics','p-value','# of lags','Num of Observations']
    
    for value,label in zip(result,labels):
        print(label+ " : "+str(value))
        
    if result[1] <= 0.05:
        print("Strong evidence against null hypothesis")
        print("reject null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis")
        print("Fail to reject null hypothesis")
        print("Data has a unit root and is non-stationary")

adf_check(df['Milk in pounds for Cow'])

df['First Difference'] = df['Milk in pounds for Cow'] - df['Milk in pounds for Cow'].shift(1)

df['First Difference'].plot()

adf_check(df['First Difference'].dropna())

#So we successfully made our time series stationary. If this wasn't enough, we would have done a second differencing on this.

#Seasonal Difference:

df['Seasonal Difference'] = df['Milk in pounds for Cow'] - df['Milk in pounds for Cow'].shift(12)

df['Seasonal Difference'].plot()

adf_check(df['Seasonal Difference'].dropna())

df['Seasonal First Difference'] = df['First Difference'] - df['First Difference'].shift(12)

df['Seasonal First Difference'].plot()

adf_check(df['Seasonal First Difference'].dropna())

#Step 4: We will now create ACF and PACF plots.

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig_first = plot_acf(df['First Difference'].dropna())

#This is an example of gradual decline.

fig_seasonal_first = plot_acf(df['Seasonal First Difference'].dropna())

#There is a sharp drop off and this is a popular auto correlation plot to see.

#We can do this using pandas also, but only for acf and not pacf

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df['Seasonal First Difference'].dropna())

result = plot_pacf(df['Seasonal First Difference'].dropna())

#We will create our final acf and pacf plots for referring to our ARIMA model:

plot_acf(df['Seasonal First Difference'].dropna());
plot_pacf(df['Seasonal First Difference'].dropna());

#Step 5: Now since our data is seasonal, we will apply seasonal ARIMA.

from statsmodels.tsa.arima_model import ARIMA

help(ARIMA)

model = sm.tsa.statespace.SARIMAX(df['Milk in pounds for Cow'],order=(0,1,0),seasonal_order=(1,1,1,12))

results = model.fit()

print(results.summary())

results.resid.plot()

results.resid.plot(kind='kde')

df['forecast'] = results.predict(start=150, end=168)
df[['Milk in pounds for Cow','forecast']].plot(figsize=(12,8))

df.tail()

from pandas.tseries.offsets import DateOffset

future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1,24)]

future_dates

future_df = pd.DataFrame(index=future_dates,columns=df.columns)

future_df

final_df = pd.concat([df,future_df])

final_df.tail()

final_df['forecast'] = results.predict(starts=168,end=192)

final_df.tail()

final_df[['Milk in pounds for Cow','forecast']].plot(figsize=(12,8))




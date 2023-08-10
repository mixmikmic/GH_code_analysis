import pandas as pd
from fbprophet import Prophet

datecol = 'Pickup_date'
df = pd.read_csv('data/uber-raw-data-janjune-15.csv', parse_dates=[datecol])

print('Number of data points =', len(df))

df.head(2)

# Create a new column for aggretating total number of trips by calendar day.
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
# for more information about Pandas 'offset aliases' for aggregating timeseries.

aggregationcol = 'num_trips'
df[aggregationcol] = pd.Series(1, index=df.index)

df_agg = df.groupby(pd.Grouper(key=datecol, freq='D')).sum()

print('Number of data points =', len(df_agg))
df_agg.head()

# Set up a dataframe in the format that Prophet expects it.
# ds is the time series, y is the variable to be predicted.

df_agg = df_agg.reset_index()
df_agg = df_agg[[datecol, aggregationcol]]
df_agg.columns = ['ds', 'y']

# Set up a Prophet model including holidays.

# Federal holidays from http://www.officeholidays.com/countries/usa/2015.php

federalholidays = pd.DataFrame({
  'holiday': 'federal',
  'ds': pd.to_datetime(['2015-01-01', '2015-01-19', '2015-05-26',
                        '2015-07-03', '2015-09-03', '2015-11-26',
                        '2015-12-25']),
  'lower_window': 0,
  'upper_window': 1,
})

m = Prophet(holidays=federalholidays, yearly_seasonality=False)

# Fit Prophet model to data.

m.fit(df_agg)

# Set up a dataframe for forecasting the timeseries.

future = m.make_future_dataframe(periods=120)
future.tail()

# Make timeseries predictions.

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Plot timeseries

m.plot(forecast)

# Plot the components of the prediction.

m.plot_components(forecast)

# Second model including uncertainties in the weekly component
# and holidays, sampled via MCMC.

model = Prophet(mcmc_samples=500, holidays=federalholidays, yearly_seasonality=False)

model.fit(df_agg)

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)
model.plot(forecast)

model.plot_components(forecast)




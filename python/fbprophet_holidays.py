import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
 
get_ipython().run_line_magic('matplotlib', 'inline')
 
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')

sales_df = pd.read_csv('../examples/retail_sales.csv', index_col='date', parse_dates=True)

sales_df.head()

df = sales_df.reset_index()

df.head()

df=df.rename(columns={'date':'ds', 'sales':'y'})

df.head()

df.set_index('ds').y.plot()

promotions = pd.DataFrame({
  'holiday': 'december_promotion',
  'ds': pd.to_datetime(['2009-12-01', '2010-12-01', '2011-12-01', '2012-12-01',
                        '2013-12-01', '2014-12-01', '2015-12-01']),
  'lower_window': 0,
  'upper_window': 0,
})

promotions

df['y'] = np.log(df['y'])

df.tail()

model = Prophet(holidays=promotions)
model.fit(df);

future = model.make_future_dataframe(periods=24, freq = 'm')
future.tail()

forecast = model.predict(future)

forecast.tail()

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

model.plot(forecast);

model.plot_components(forecast);

model_no_holiday = Prophet()
model_no_holiday.fit(df);

future_no_holiday = model_no_holiday.make_future_dataframe(periods=24, freq = 'm')
future_no_holiday.tail()

forecast_no_holiday = model_no_holiday.predict(future)

forecast.set_index('ds', inplace=True)
forecast_no_holiday.set_index('ds', inplace=True)
compared_df = forecast.join(forecast_no_holiday, rsuffix="_no_holiday")

compared_df= np.exp(compared_df[['yhat', 'yhat_no_holiday']])

compared_df['diff_per'] = 100*(compared_df['yhat'] - compared_df['yhat_no_holiday']) / compared_df['yhat_no_holiday']
compared_df.tail()

compared_df['diff_per'].mean()




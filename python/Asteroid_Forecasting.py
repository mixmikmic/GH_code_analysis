import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib

df = pd.read_csv('/Data')
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')

df.head()



model = Prophet()
model.fit(df);

future = model.make_future_dataframe(periods=30)
future

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

get_ipython().magic('matplotlib inline')
c = model.plot(forecast);

model.plot_components(forecast);




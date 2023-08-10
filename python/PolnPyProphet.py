import pandas as pd
import numpy as np
from fbprophet import Prophet

df = pd.read_csv('../analyze/pollen_weather.csv')
df = df[['Date', 'Alnus']]
df = df.rename(columns={'Date': 'ds', 'Alnus': 'y'})
df['cap'] = 400
# df.head()
df.tail()

# m = Prophet(growth='logistic')
m = Prophet()
m.fit(df);

future = m.make_future_dataframe(periods=3)
future.tail()
# future = m.make_future_dataframe(periods=3)
# future['cap'] = 400

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# forecast = m.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast);

# Python
# m.plot(forecast);

m.plot_components(forecast);




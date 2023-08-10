get_ipython().magic('matplotlib inline')
from fbprophet import Prophet

import pandas as pd

# Read the SPY_notional.csv file
# Prophet expects the days/time to be in the 'ds' column and the values in the 'y' column.
df = pd.read_csv('SPY_notional.csv')
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast);

m.plot_components(forecast);

# Filter the DataFrame
df = df[df['ds'] >= '2010-01-01']

# Create and fit a new Prophet model
m = Prophet()
m.fit(df)

# Make the predictions
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Plot
m.plot(forecast);

# Let's do this quickly for another ETF - VTI
df = pd.read_csv('VTI_notional.csv')

# Filter the DataFrame
df = df[df['ds'] >= '2010-01-01']

# Create and fit a new Prophet model
m = Prophet()
m.fit(df)

# Make the predictions
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Plot
m.plot(forecast);




import pandas as pd
import numpy as np
from fbprophet import Prophet 
import matplotlib 

#Load the dataset . ds column contains the dates for wich a prediction is to be made. 
df = pd.read_csv('Data.csv')
df['ds'] = pd.to_datetime(df['ds'], format= '%Y-%m-%d')

df.head()
#we order the dataset by date 
df.sort_values(by='ds')

len(df)
data_10_years = df[0:5200]
data_10_years

#We fit the model by instantiationg a new Prophet object. 
model = Prophet()
model.fit(data_10_years);

#This means that we don't settle a pattern during the day 

# Made a suitable dataframe that extends into the future in a specified number of days using the helper method bellow 
future = model.make_future_dataframe(periods=50)
future.tail()

# The predict method will asign each row in future a predicted value :
# yhat 

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#Python
get_ipython().magic('matplotlib inline')
model.plot(forecast);

## Forecast components. Plotted the trends, yearly and weekly seasonality
model.plot_components(forecast);

## Asteoid Detection in one year time 


data_1_year = df[454:1407]

model_1_year = Prophet(yearly_seasonality=True)
model_1_year.fit(data_1_year)
future_1_year = model_1_year.make_future_dataframe(periods=30)

forecast_1_year = model_1_year.predict(future_1_year)
get_ipython().magic('matplotlib inline')
model_1_year.plot(forecast_1_year)

data_6_month = df[0:454]

model_6_month = Prophet(yearly_seasonality=True)
model_6_month.fit(data_6_month)
future_6_month = model_6_month.make_future_dataframe(periods=30)

forecast_6_month = model_6_month.predict(future_6_month)
get_ipython().magic('matplotlib inline')
model_1_year.plot(forecast_6_month)




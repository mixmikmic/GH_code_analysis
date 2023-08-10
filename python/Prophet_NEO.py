import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib

df = pd.read_csv('/home/gparreno/DataScience/Deep_Asteroid/Prophet3.csv')
#For the model to fit, you have to name the variable assigned to date with ds,y name 
#df['y'] = np.log(df['y'])

df

type(df['ds'][0])

#pd.to_datetime(df['ds'], format='%Y,%m,%d')

model = Prophet()
model.fit(df);

future = model.make_future_dataframe(periods=30)
future

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

get_ipython().magic('matplotlib inline')
c = model.plot(forecast);

print(c)

c.show()

c




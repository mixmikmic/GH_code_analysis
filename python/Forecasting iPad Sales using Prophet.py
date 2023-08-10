from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

ds = ['6/21/2010','9/20/2010','12/27/2010',
      '3/23/2011','6/22/2011','9/21/2011','12/28/2011',
      '3/23/2012','6/22/2012','9/21/2012','12/28/2012',
      '3/24/2013','6/23/2013','9/22/2013','12/29/2013',
      '3/25/2014','6/24/2014','9/23/2014','12/30/2014',
      '3/26/2015','6/25/2015','9/24/2015','12/31/2015',
      '3/26/2016','6/25/2016','9/24/2016','12/31/2016']
y = [3.27, 4.19, 7.33,
     4.69, 9.25, 11.12, 15.43,
     11.8, 17.04, 14.04, 22.86,
     19.48, 14.62, 14.08, 26.04,
     16.35, 13.28, 12.32, 21.42,
     12.62, 10.93, 9.88, 16.12,
     10.25, 9.95, 9.27, 13.08]
df = pd.DataFrame(data={'ds': ds, 'y': y})

ds = list(map(mdates.strpdate2num('%m/%d/%Y'), df['ds']))
plt.plot_date(ds, df['y'])

df['y'] = np.log(df['y'])

h = pd.DataFrame({
    'holiday': 'release',
    'ds': ['4/3/2010', '3/11/2011', '11/2/2012', '11/1/2013', '10/22/2014', '3/24/2017'],
    'lower_window': 0.0,
    'upper_window': 366/4
  })

m = Prophet(growth='logistic', holidays=h)
cap = 1.1 * np.max(df['y'])
df['cap'] = cap
fit = m.fit(df)

future = m.make_future_dataframe(periods=4, freq='BQ')
future['cap'] = cap
forecast = fit.predict(future)

m.plot(forecast)

m.plot_components(forecast)


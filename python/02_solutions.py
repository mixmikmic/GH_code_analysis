import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

get_ipython().run_line_magic('matplotlib', 'inline')

AIR_QUALITY_URL = 'https://files.datapress.com/london/dataset/london-average-air-quality-levels/2017-10-09T08:41:22.63/air-quality-london-monthly-averages.csv'

aq = pd.read_csv(AIR_QUALITY_URL)

aq['Month'] = pd.to_datetime(aq['Month'], format='%b-%y')
aq.set_index('Month', inplace=True)

aq = aq.resample('M').sum()

pm25 = aq.loc[:,'London Mean Roadside:PM2.5 Particulate (ug/m3)'].rename('PM2.5')

pm25.isnull().sum()

pm25.fillna(method='pad', inplace=True)

pm25.plot()

plot_acf(pm25, lags=12)

plot_pacf(pm25, lags=12)

pm25_data = pd.DataFrame({
    'pm25': pm25,
    'pm25_shift1': pm25.shift(1),
    'pm25_shift12': pm25.shift(12)
}).dropna()

model1 = smf.glm('pm25 ~ pm25_shift1 + pm25_shift12', data=pm25_data).fit()

model1.summary()

plot_acf(model1.resid_response, lags=12)

plot_pacf(model1.resid_response, lags=12)

pm25.plot()
model1.fittedvalues.plot()

model2 = sm.tsa.ARIMA(pm25, (1, 0, 1)).fit()

model2.summary()

plot_acf(model2.resid, lags=12)

plot_pacf(model2.resid, lags=12)

model2.plot_predict('2015-01-31', '2017-12-31')


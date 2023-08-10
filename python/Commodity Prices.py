get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dfCoal = pd.read_csv('./data/raw/Quantdl - Coal (Daily, US)/DOE-coal.csv')
dfCoal.columns = ['date', 'Central Appalachia', 'Northern Appalachia', 'Illinois Basin', 'Powder River Basin', 'Uinta Basin']
dfCoal = dfCoal.set_index('date').sort_index()
dfCoal.index = pd.to_datetime(dfCoal.index)
dfCoal.head()

dfNg = pd.read_csv('../data/raw/EIA - Natural gas henry hub daily prices/Natural gas henry hub daily prices.csv', 
                   parse_dates=['Date'])
dfNg.columns = ['date', 'NgPrice']
dfNg = dfNg.set_index('date').sort_index()
dfNg.index = pd.to_datetime(dfNg.index)
dfNg.head()

dfNg.to_csv('../data/NGPrices.csv')

dfCommodity = dfCoal.merge(dfNg, left_index=True, right_index=True)
dfCommodity.plot(figsize=(20,10))
plt.xlabel('Year')

dfCommodity = dfCommodity.resample('D', fill_method=None).interpolate()
dfCommodity.head()

dfCommodity.to_csv('./data/commodityPrices.csv')


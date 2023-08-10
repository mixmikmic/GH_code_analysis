get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 20,10

# Commodity Prices
dfCommodity = pd.read_csv('../data/commodityPrices.csv')
dfCommodity['date'] = pd.to_datetime(dfCommodity['date'])
dfCommodity = dfCommodity.set_index('date').sort_index()

# Wind Generation
dfWind = pd.read_csv('../data/MISOWindGeneration.csv')
dfWind['date'] = pd.to_datetime(dfWind['date'])
dfWind = dfWind.set_index('date').sort_index()

# Demand
dfLoad = pd.read_csv('../data/MISOActualLoad.csv')
dfLoad['Market Day'] = pd.to_datetime(dfLoad['Market Day'])
dfLoad = dfLoad.set_index('Market Day').sort_index()
dfLoad.index.names = ['date']
dfLoadActual = dfLoad[['Central ActualLoad (MWh)', 'East ActualLoad (MWh)', 'MISO ActualLoad (MWh)', 'Midwest ISO ActualLoad (MWh)', 'North ActualLoad (MWh)', 'South ActualLoad (MWh)', 'West ActualLoad (MWh)']]
dfLoadActual = dfLoadActual.fillna(0)    # Handle NaN

# Merge into a single DataFrame
dfX = pd.merge(dfCommodity, dfWind, left_index=True, right_index=True)
dfX = pd.merge(dfX, dfLoadActual, left_index=True, right_index=True)
dfX.head()

#LMP
#dfMiso = pd.read_hdf('../data/LMP.h5','lmp')
dfMiso = pd.read_hdf('../data/LMP-ACEI_AMMO.h5','lmp')
dfMiso.index = pd.to_datetime(dfMiso.index)

tsY = dfMiso['meanPrice']     # converted to Pandas.Series
tsY.index = pd.to_datetime(tsY.index)
dfY = pd.DataFrame(tsY)

# Combine X and Y
df = pd.merge(dfY, dfX, left_index=True, right_index=True, how='inner')
df = df[:'2013-09-01']
df.plot()

plt.scatter(df['NgPrice'], df['meanPrice'])
plt.xlabel('Natural Gas Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Natural Gas')

plt.scatter(np.log(df['NgPrice']), np.log(df['meanPrice']))
plt.xlabel('Natural Gas Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Natural Gas (Log Transformed)')

df2 = df[df['meanPrice']>15]
plt.scatter(np.log(df2['NgPrice']), np.log(df2['meanPrice']))
plt.xlabel('Natural Gas Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Natural Gas (Log Transformed, LMP>15)')

from scipy import stats


df2 = df[df['meanPrice']>15]
df2 = df2[(np.abs(stats.zscore(df2['meanPrice'])) < 2)]

plt.scatter(np.log(df2['NgPrice']), np.log(df2['meanPrice']))
plt.xlabel('Natural Gas Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Natural Gas (Log Transformed, LMP>15, LMP<3sd)')

fig = plt.figure(figsize=(20,30))

ax1 = fig.add_subplot(321)
plt.scatter(df['Central Appalachia'], df['meanPrice'])
plt.xlabel('Coal Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Coal Prices (Central Appalachia)')

ax1 = fig.add_subplot(322)
plt.scatter(df['Northern Appalachia'], df['meanPrice'])
plt.xlabel('Coal Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Coal Prices(Northern Appalachia)')

ax1 = fig.add_subplot(323)
plt.scatter(df['Illinois Basin'], df['meanPrice'])
plt.xlabel('Coal Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Coal Prices(Illinois Basin)')

ax1 = fig.add_subplot(324)
plt.scatter(df['Powder River Basin'], df['meanPrice'])
plt.xlabel('Coal Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Coal Prices(Powder River Basin)')

ax1 = fig.add_subplot(325)
plt.scatter(df['Uinta Basin'], df['meanPrice'])
plt.xlabel('Coal Prices')
plt.ylabel('LMP')
plt.title('LMP vs. Coal Prices(Uinta Basin)')




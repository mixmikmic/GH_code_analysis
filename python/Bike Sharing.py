path = 'C:/Users/Marie/Desktop/Bike Sharing'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from os import listdir
from sklearn.preprocessing import MinMaxScaler

print(listdir(path + '/Data'))

data_day = pd.read_csv(path + '/Data/day.csv')

data_hour = pd.read_csv(path + '/Data/hour.csv')

data_day.dteday = pd.to_datetime(data_day.dteday)
data_hour.dteday = pd.to_datetime(data_hour.dteday)

data_day['casual_scaled'] = 0

data_day['registered_scaled'] = 0

data_day['cnt_scaled'] = 0

data_day[['casual_scaled', 'registered_scaled', 'cnt_scaled']] = MinMaxScaler().fit_transform(data_day[['casual', 'registered', 'cnt']])

data_day = data_day.set_index('dteday', drop = True)

for col in ['registered', 'casual', 'cnt']:
    data_day[col].plot(figsize = (16,8))
    plt.show()

data_day.head()

def plot_aggregated_data(data = data_day, aggregated_variable = 'weekday', features = ['casual', 'registered', 'cnt']):
    """
    Plotting aggregated bar charts
    """
    aggregation = {i:[min, max, np.mean, np.median] for i in features}
    
    agg_data = data.groupby(aggregated_variable).agg(aggregation)
    
    for ft in features:
        agg_data[ft].plot(kind = 'bar', figsize = (16,8), title = ft)
        plt.show()

plot_aggregated_data(data = data_hour, aggregated_variable='weathersit')

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,8))
ax1.scatter(data_day.temp, data_day.casual_scaled, color = 'orange')
ax1.set_title('Casual Users')
ax2.scatter(data_day.temp, data_day.registered_scaled, color = 'green')
ax2.set_title('Registered Users')
plt.show()

plt.figure(figsize = (16,8))
plt.scatter(data_day.temp, data_day.casual_scaled, c = 'orange')
plt.scatter(data_day.temp, data_day.registered_scaled, c = 'green')
plt.legend(['Casual Users', 'Registered Users'])
plt.title('Temperature')
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,8))
ax1.scatter(data_day.hum, data_day.casual, color = 'orange')
ax1.set_title('Casual Users')
ax2.scatter(data_day.hum, data_day.registered, color = 'green')
ax2.set_title('Registered Users')
plt.show()

plt.figure(figsize = (16,8))
plt.scatter(data_day.hum, data_day.casual_scaled, c = 'orange')
plt.scatter(data_day.hum, data_day.registered_scaled, c = 'green')
plt.legend(['Casual Users', 'Registered Users'])
plt.title('Humidity')
plt.show()

data_day['hum_bin'] = ((data_day.hum *20).astype(int)/20.)

agg_hum = data_day[['hum_bin', 'casual', 'registered', 'cnt']].groupby('hum_bin').sum()
agg_hum.plot(kind = 'bar', figsize = (16,8))
#for col in agg_hum.columns :
#    agg_hum[col].plot(kind = 'bar', figsize = (16,8))
#    plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(data_day[['temp', 'hum']], day['cnt'])

plt.scatter(model.predict(data_day[['temp', 'hum']]), day['cnt'])
plt.show()

np.corrcoef(data_day['casual'], data_day['registered'])

data_day.head()




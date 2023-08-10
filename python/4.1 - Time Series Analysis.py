import pandas as pd

fname = 'AirPassengers.csv'

data = pd.read_csv(fname)

data.head()

data.isnull().sum()

data.dtypes

data['Month'] = pd.to_datetime(data['Month'])
data.head()

data.dtypes

data['Month'].dt.year.head()

data = data.set_index('Month')
data.head()

get_ipython().run_line_magic('matplotlib', 'inline')

data.plot(grid='on')

from datetime import datetime

start_date = datetime(1959, 1, 1)
end_date = datetime(1960, 12, 1)
data[(start_date <= data.index) & (data.index <= end_date)].plot(grid='on')

import statsmodels.api as sm

decomposition = sm.tsa.seasonal_decompose(data, model='additive')
fig = decomposition.plot()

# decomposition.plot()  # if using outside notebook

import matplotlib

matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]  # double up default plot size

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ax.grid(True)

year = mdates.YearLocator(month=1)
month = mdates.MonthLocator(interval=3)
year_format = mdates.DateFormatter("%Y")
month_format = mdates.DateFormatter("%m")

ax.xaxis.set_minor_locator(month)

ax.xaxis.grid(True, which='minor')
ax.xaxis.set_major_locator(year)
ax.xaxis.set_major_formatter(year_format)

plt.plot(data.index, data['AirPassengers'], c='blue')
plt.plot(decomposition.trend.index, decomposition.trend, c='red')




get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
mpl.rcParams['figure.figsize'] = (12.5, 6.0)
import pandas as pd
pd.options.display.max_rows = 12

country = ['Northern Ireland', 'Scotland', 'Wales', 'England', 'Isle of Man']
capital = ['Belfast', 'Edinburgh', 'Cardiff', 'London', 'Douglas']
area = [14130, 77933, 20779, 130279, 572]
population2017 = [1876695, 5404700, np.nan, 55268100, np.nan]
population2011 = [1810863, 5313600, 3063456, 53012456, 83314]

series = pd.Series(area)
series

uk_area = pd.Series(area, index=country)
uk_area

uk_area.values, uk_area.values.dtype, uk_area.index

uk_area['Wales'], uk_area[2], uk_area.values[2]

uk_area[0:3]

uk_area[['Wales', 'Scotland']]

uk_area['England':'Scotland']  # oops

uk_area.sort_index(inplace=True)
uk_area['England':'Scotland']

uk_area

uk_area['England':'Scotland']  # Inclusive!

uk_area[0:3]  # Exclusive!

series.index = [1, 2, 3, 4, 5]
series

series[1], series.loc[1], series.iloc[1]

list(series[1:3]), list(series.loc[1:3]), list(series.iloc[1:3])

uk_area[uk_area > 20000]

uk_area * 0.386  # convert to square miles (1/1.61**2)

uk_area.sum()  # Isle of Man is not part of the UK!  We'll fix that later.

p11 = pd.Series(population2011, index=country)
p17 = pd.Series(population2017, index=country).dropna()  # disregard nulls, we will see more later

p11

p17

p17 - p11

array = np.array([area, capital, population2011, population2017]).T  # transpose
data = pd.DataFrame({'capital': capital,
                     'area': area,
                     'population 2011': population2011,
                     'population 2017': population2017},
                    index=country)

array

data

array[0]

data['area']  # get column

data.iloc[0]  # but `iloc` does the same as a `NumPy` array

data.area  # this works too

data.loc['England', 'area']  # still [row, column]

data.sort_index(inplace=True)

data.head(3)

data.sort_values('area').tail(3)

len(data)  # number of rows

data.describe()

data.info()

plot = data[['population 2011', 'population 2017']].plot(kind='bar')
ticks = ['%.0f M' % (x[1] / 1e6) for x in plot.yaxis.iter_ticks()]
plot.yaxis.set_ticklabels(ticks);  # just a hack to get nice ticks

plot = data.plot(kind='scatter', x='population 2011', y='area', loglog=True)
for k, v in data[['population 2011', 'area']].iterrows():
    plot.axes.annotate(k, v)

data['capital'].str.startswith('Be')

data[data.capital.str.contains('[oa]')]  # regex

data[data.index.str.startswith('Eng')]

data.dropna()  # We lost Wales and the Isle of Man!

data.dropna(axis='columns')  # that's better

data_full = data.fillna(method='ffill', axis='columns')
data_full

data_full.dtypes

data_full = data_full.apply(pd.to_numeric, errors='ignore')
data_full = data_full.astype(np.integer, errors='ignore')
data_full.dtypes

data_full


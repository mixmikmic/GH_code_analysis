import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

get_ipython().magic('matplotlib inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

address = 'mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
cars.index = cars.car_names
mpg = cars['mpg']

mpg.plot(kind='hist')

plt.hist(mpg)
plt.plot()

sb.distplot(mpg)

cars.plot(kind='scatter', x='hp', y='mpg', c=['darkgray'], s=150)

sb.regplot(x='hp', y='mpg', data=cars, scatter=True)

sb.pairplot(cars)

cars_df = pd.DataFrame((cars.ix[:,(1,3,4,6)].values), columns = ['mpg', 'disp', 'hp', 'wt'])
cars_target = cars.ix[:,9].values
target_names = [0, 1]

cars_df['group'] = pd.Series(cars_target, dtype="category")
sb.pairplot(cars_df, hue='group', palette='hls')



cars.boxplot(column='mpg', by='am')
cars.boxplot(column='wt', by='am')

sb.boxplot(x='am', y='mpg', data=cars, palette='hls')




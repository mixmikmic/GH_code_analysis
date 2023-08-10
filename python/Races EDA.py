import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from __future__ import division
get_ipython().magic('matplotlib inline')

race = pd.read_csv('races.csv')
race.head(2)

race.info()

race['config'].value_counts()

race['surface'].value_counts()

race['distance'].value_counts()

race['going'].value_counts()

race['horse_ratings'].value_counts()

race['place_combination1'].hist(bins = np.linspace(0.5,15.5,16))
plt.show()

race.ix[race['win_combination1'] != race['place_combination1'],['place_combination1','place_combination2','place_combination3']]

race[race['win_combination1'] != race['place_combination1']]['win_combination1']

race[race['win_combination1'] != race['place_combination1']]['place_combination2']



race[['place_combination1','place_combination2']].hist(bins=np.linspace(0.5,14.5,15))

place_lst = pd.concat([race['place_combination1'], race['place_combination2'], race['place_combination3']])

fig = plt.figure(figsize=[10,7])
ax = fig.add_subplot(111)
ax.hist(place_lst, bins=np.linspace(0.5,14.5,15))
plt.show()

all(place_lst.notnull())

place_lst.dropna(inplace=True)

pd.pivot_table(race, index=['distance'], values=['time1'], aggfunc=np.median)

race[race['distance']==1200]['time3']

pd.pivot_table(race, index=['distance'], values=['win_dividend1'], aggfunc=[np.mean,np.median,np.sum])

race[race['win_dividend1']<1000]['win_dividend1'].hist()

np.percentile(race['win_dividend1'],90)




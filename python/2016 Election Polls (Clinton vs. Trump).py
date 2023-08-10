import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')
from __future__ import division

poll_df = pd.read_csv('2016-general-election-trump-vs-clinton.csv')
poll_df.head()

poll_df.info()

sns.factorplot('Affiliation',data=poll_df,kind='count')

sns.factorplot('Affiliation',data=poll_df,kind='count', order=['Rep', 'Dem'])

sns.factorplot('Affiliation',data=poll_df,kind='count',hue='Population')

avg = pd.DataFrame(poll_df.mean())
avg.drop('Number of Observations',axis=0,inplace=True)

avg

std = pd.DataFrame(poll_df.std())
std.drop('Number of Observations', axis=0, inplace=True)

std

avg.plot(yerr=std,kind='bar',legend=False)

poll_avg = pd.concat([avg,std],axis=1)
poll_avg.columns = ['Average','STD']
poll_avg

poll_df.plot(x='End Date',y = ['Clinton','Trump','Undecided'],
             linestyle='',marker='o')

from datetime import datetime
poll_df['Difference']= (poll_df['Clinton'] - poll_df['Trump'])/100
poll_df.head()

combo_dates_df = poll_df.groupby(['Start Date'],as_index=False).mean()
combo_dates_df.head()

combo_dates_df.plot('Start Date','Difference',figsize=(12,4),marker='o',
             linestyle='-',color='orange')

combo_dates_df[combo_dates_df['Difference']==combo_dates_df['Difference'].min()]

poll_df[poll_df['Start Date']=='2015-08-21']

pollster_df = poll_df.groupby(['Pollster'],as_index=False).mean()

pollster_df.head()

pollster_df['Pollster'].count()

pollster_df=pollster_df.drop(pollster_df.columns[[1,4,5,6]], axis=1)

pollster_df=pollster_df.set_index('Pollster')

pollster_df.head()

pollster_df.plot(kind='barh',figsize=(20,24),cmap='seismic')

poll_df.plot(x='End Date',y = ['Clinton','Trump'],
             linestyle='',marker='o')

poll_df=poll_df.sort('End Date')

#remove polls where number of observations is not available
poll_df=poll_df.dropna()
poll_df.head()

poll_df.plot(x='End Date',y = ['Clinton','Trump'],
             linestyle='',marker='o')




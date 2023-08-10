get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 6

# Wage dataset from https://cran.r-project.org/web/packages/ISLR/
df = pd.read_csv('./data/wage_data.csv', index_col=0)
df

pd.options.display.max_rows = 10
print(df.describe())
pd.options.display.max_rows = 6

df.columns

df['sex'].unique()

df['race'].unique()

df['race'] = df['race'].apply(lambda x: x.split('. ')[1])
df['race'].unique()

df['education'].unique()

df['education'] = df['education'].replace({'1. < HS Grad': 'Primary Education', 
                                           '2. HS Grad': 'Secondary Education', 
                                           '3. Some College': 'Postsecondary Education',
                                           '4. College Grad': 'Postsecondary Education', 
                                           '5. Advanced Degree': 'Advanced Degree'})

df['education'].unique()

from collections import Counter
Counter(df.education)

df['age_class'] = pd.cut(df['age'], [0, 18, 35, 65, np.Inf], labels=['0-18', '19-35', '36-65', '66+'])
df

df.groupby('age_class')['wage'].agg([np.mean, np.std])

df.groupby('race')['wage'].agg([np.mean, np.std])

pd.options.display.max_rows = 100
df.groupby(['age_class', 'race'])['wage'].agg([np.mean, np.std])

fig, axs = plt.subplots(nrows=2)
df['wage'].plot.hist(bins=50, ax=axs[0])
df['logwage'].plot.hist(bins=50, ax=axs[1])

# %% scatter plot wage ~ age
fig, ax = plt.subplots()
df.plot.scatter(x='age', y='wage', title='wage ~ age', alpha=0.5, label='data', ax=ax)

# compute median wage per age and plot
df.groupby('age')['wage'].agg(np.median).plot.line(ax=ax, label='median', color='g')
df.groupby('age')['wage'].agg(np.mean).plot.line(ax=ax, label='mean', color='r')
ax.legend()

df.boxplot(column='wage', by='education')

# read in data
pd.options.display.max_rows = 7
names = ','.join(['date'] + ["{:02d},flag".format(i) for i in range(24)]).split(',')
df = pd.read_csv('./data/AT90AKC0000800100hour.1-1-1988.31-12-2012',  # measured at AKH-Wien
                 sep='\t', header=None, names=names)
df

df = df.drop('flag', axis=1)
df

df = df.set_index('date')
df

# data is not in the right shape, want index to be time and then have only one column
df_stacked = df.stack()
df_stacked

df_stacked = df_stacked.reset_index(name='no2')
df_stacked

df_stacked.index = pd.to_datetime(df_stacked['date'] + df_stacked['level_1'], format="%Y-%m-%d%H")
df_stacked = df_stacked.drop(['date', 'level_1'], axis=1)

df_stacked

df_stacked.info()

# partial string indexing
df_stacked['2012-01':'2012-02'].plot()

# pandas power - resampling
df_stacked.resample('D').head()

df_stacked.resample('D')['no2']['2005':'2006'].plot.line()

# a datetimeindex knows about weekdays
df_stacked.index.weekday

df_stacked['weekday'] = df_stacked.index.weekday
df_stacked['hour'] = df_stacked.index.hour
df_stacked

# introduce new variable to compare emissions on weekends
df_stacked['is_weekend'] = df_stacked['weekday'].isin([5, 6])
df_stacked

df_agg = df_stacked.groupby(['is_weekend', 'hour'], as_index=False).agg(np.median)
df_agg

fig, ax = plt.subplots()
df_agg[df_agg['is_weekend'] == True].plot.line(x='hour', y='no2', ax=ax, label='weekend')
df_agg[df_agg['is_weekend'] == False].plot.line(x='hour', y='no2', ax=ax, label='not weekend')


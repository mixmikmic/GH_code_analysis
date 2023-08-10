import numpy as np
import pandas as pd

df = pd.read_csv('nycflights.csv')

print(df.head())

df.info()

df.shape

df['origin'].unique()

df['year'].astype(str) + "-" + df['month'].astype(str) + '-' + df['day'].astype(str)

df['date'] = df['year'].astype(str) + "-" + df['month'].astype(str) + "-" + df['day'].astype(str)

print(df['date'][:10])

df['date'] = pd.to_datetime(df['date'])

df.info()

df = df.set_index(['date'])

df.index

print(df.head())

df.groupby('carrier')

df.groupby('carrier')['dest']

df.groupby('carrier')['dest'].unique()

df.groupby('carrier')['origin'].unique()

# busiest origin?
print(df.groupby('origin')['dep_time'].count())  # has NaNs, and they are not counted
print(df.groupby('origin')['dest'].count()) # no NaNs

print(df.groupby('month')['dep_delay'].mean())

# how many flights to LAX?
sum(df['dest'] == 'LAX')

lax_df = df[ df['dest']=='LAX' ]

lax_df.head()

lax_df.groupby('origin')['origin'].count()

print(lax_df.groupby('origin')['dep_delay'].median())
print(lax_df.groupby('origin')['dep_delay'].mean())

print(lax_df.groupby('carrier')['dep_delay'].median())
print(lax_df.groupby('carrier')['dep_delay'].mean())

print(lax_df.groupby(['carrier','origin'])['dep_delay'].mean())
print(lax_df.groupby(['carrier','origin'])['dep_delay'].median())

# all flights again

# on what days was the departure delay the largest on average?
df.groupby('date')['dep_delay'].mean().sort_values(ascending = False).head()

# on what days did we have the greatest number of delayed flights?
df[df['dep_delay'] > 0].groupby('date')['dep_delay'].count().sort_values(ascending = False).head()

# subset the dataframe to rows where dep-delay is > 0
# groupby month and carrier
# select the dep_delay column
# we count those
df[ df['dep_delay'] > 0 ].groupby(['month','carrier'])['dep_delay'].aggregate('count').unstack()
# number of flights for each month and each carrier that were delayed

# number of flights total by month and carrier
df.groupby(['month','carrier'])['dep_time'].aggregate('count').unstack()

df.pivot_table(index = 'month', columns = 'carrier', values = 'dep_time', aggfunc = 'count')

df[ df['dep_delay'] > 0].pivot_table(index = 'month', columns = 'carrier', values = 'dep_time', aggfunc = 'count') / df.pivot_table(index = 'month', columns = 'carrier', values = 'dep_time', aggfunc = 'count')

df[df['dep_delay'] > 0].groupby('month')['dep_delay'].count() / df.groupby('month')['dep_delay'].count()

( df[df['arr_delay'] > 20].groupby('carrier')['arr_delay'].count() / df.groupby('carrier')['arr_delay'].count() ).sort_values(ascending = False)


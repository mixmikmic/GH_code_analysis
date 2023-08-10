import pandas as pd

df = pd.read_csv('../data/eels.csv')

df.head()

df[['country', 'kilos']].groupby('country').sum().sort_values('kilos', ascending=False)

df.country.value_counts().sort_values(ascending=False).head()

df[df['kilos'] > 25000].country.value_counts().sort_values(ascending=False)

df[['country', 'year', 'kilos']].groupby(['country', 'year']).sum()

pd.pivot_table(df,
               index='country',
               columns='year',
               values='kilos',
               aggfunc=sum).sort_values(2017, ascending=False) \
                           .fillna(0)




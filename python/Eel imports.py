import pandas as pd

df = pd.read_csv('../data/eels.csv')

df.head()

df.info()

df.year.unique()

df.month.unique()

df.country.unique()

# have to use bracket notation bc "product" is a pandas function
df['product'].unique()

print(df.kilos.max())
print(df.kilos.min())

print(df.dollars.max())
print(df.dollars.min())

for yeargroup in df[['year', 'month']].groupby('year'):
    print(yeargroup[0], yeargroup[1].month.unique())

df[['country', 'kilos']].groupby('country')                         .sum()                         .sort_values('kilos', ascending=False)

pivoted_sums = pd.pivot_table(df,
                              aggfunc='sum',
                              values='kilos',
                              index='country',
                              columns='year')

pivoted_sums.sort_values(2017, ascending=False)

pivoted_sums_notnull = pivoted_sums[pivoted_sums[2010].notnull()]

pivoted_sums_notnull

pivoted_sums_notnull['10to16pctchange'] = (pivoted_sums_notnull[2016] - pivoted_sums_notnull[2010]) / pivoted_sums_notnull[2010]

pivoted_sums_notnull.sort_values('10to16pctchange')

pop_products = df[['product', 'kilos']].groupby('product')                                        .sum()                                        .sort_values('kilos', ascending=False)

pop_products


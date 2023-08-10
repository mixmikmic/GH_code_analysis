import pandas as pd

country_codes = pd.read_csv('../data/country-codes.csv', dtype={'code': str})

country_codes.head()

country_pop = pd.read_csv('../data/country-population.csv', dtype={'code': str})

country_pop.head()

country_pop['pct_change'] = ((country_pop['pop2015'] - country_pop['pop2000']) / country_pop['pop2000']) * 100

country_pop.head()

top_change = country_pop.sort_values('pct_change', ascending=False)[['code', 'pop2000', 'pop2015', 'pct_change']]

top_change.head()

merged = pd.merge(top_change, country_codes, on='code')

merged.head()


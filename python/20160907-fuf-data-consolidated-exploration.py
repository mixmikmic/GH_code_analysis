import pandas as pd, numpy as np

fuf_data = pd.read_excel('../data/fuf_consolidated_thru_8_22.xlsx', sheetname=0)
fuf_data_updated = pd.read_csv('../data/combined_tree_data_with_header.csv', header = False)

fuf_data.info()

fuf_data_updated.info()

data = fuf_data[['ON_ADR','ONSTREET', 'PROPSTREET', 'PROP_ADR']]
data.count()

data = fuf_data[['ON_ADR','ONSTREET', 'PROPSTREET', 'PROP_ADR']]
data.ix[data['ONSTREET'] != data['PROPSTREET']].head()

fuf_data.describe()

fuf_data['HARDSCAPE'].describe()

fuf_data['HARDSCAPE'].value_counts()

fuf_data['PROPERTY'].value_counts()

fuf_data['CONDITION'].value_counts()

fuf_data['HARDSCAPE_BINARY'] = fuf_data['HARDSCAPE'].apply(lambda x: 0 if x == 'None' else 1)
fuf_data['VACANT_LOT'] = fuf_data['CONDITION'].apply(lambda x: 0 if x != 'Vacancy' else 1)
fuf_data['CONDITION_BINARY'] = fuf_data['CONDITION'].apply(lambda x: 0 if x in ['Poor','Dead',
                                                                               'Critical'] else 1)
fuf_data['STUMP'] = fuf_data['CONDITION'].apply(lambda x: 1 if x in ['Stump', 'Stump Removal'] else 0)
grouped = fuf_data[['HARDSCAPE_BINARY','EXACT_DBH','PROPERTY',
          'VACANT_LOT', 'CONDITION_BINARY', 'STUMP']].groupby('PROPERTY').mean().reset_index()
joined = pd.merge(fuf_data, grouped, on='PROPERTY', how = 'left', suffixes=('_binary','_perc'))

grouped

joined.columns

joined.head()

joined.info()

joined.to_csv("fuf_with_metrics.csv")




import pandas as pd, numpy as np

fuf_data_updated = pd.read_csv('../data/combined_tree_data_with_header.csv', header = False)

fuf_data_updated.info()

# Dead, Vacancy, Poor, Stump, Stump Removal, Unsuitable Site

fuf_data_updated.condition.value_counts()

fuf_data_updated[(fuf_data_updated.condition.str.contains('Vacancy')==True)]

fuf_data_updated.describe()

fuf_data_updated['hardscape_damage'].describe()

fuf_data_updated['hardscape_damage'].value_counts()

fuf_data_updated['neighborhood'].value_counts()

fuf_data_updated['condition'].value_counts()

fuf_data_updated['diameter_at_breast_height'].value_counts()

fuf_data_updated['condition'].value_counts()

# Dead, Vacancy, Poor, Stump, Stump Removal, Unsuitable Site

fuf_data_updated['hardscape_metric'] = fuf_data_updated['hardscape_damage'].apply(lambda x: 0 if x in ['None',
                                                                                    'No', 'NA'] else 1)
fuf_data_updated['vacant_lot_metric'] = fuf_data_updated['condition'].apply(lambda x: 0 if x != 'Vacancy' else 1)
fuf_data_updated['condition_metric'] = fuf_data_updated['condition'].apply(lambda x: 0 if x in ['Poor','Dead',
                                                                               'Critical','Stump','Stump Removal',
                                                                                'Unsuitable Site'] else 1)
fuf_data_updated['stump_metric'] = fuf_data_updated['condition'].apply(lambda x: 1 if x in ['Stump', 
                                                                                            'Stump Removal'] else 0)
grouped = fuf_data_updated[['hardscape_metric','diameter_at_breast_height','neighborhood',
          'vacant_lot_metric', 'condition_metric', 'stump_metric']].groupby('neighborhood').mean().reset_index()
joined = pd.merge(fuf_data_updated, grouped, on='neighborhood', how = 'left', suffixes=('_binary','_perc'))

grouped

grouped_with_count = fuf_data_updated[['hardscape_metric','diameter_at_breast_height','neighborhood',
          'vacant_lot_metric', 'condition_metric', 'stump_metric']].groupby('neighborhood').agg(['mean',
                                                                                                'count']).reset_index()
grouped_with_count

joined.columns

joined.head()

joined.info()

joined.to_csv("fuf_with_metrics.csv")




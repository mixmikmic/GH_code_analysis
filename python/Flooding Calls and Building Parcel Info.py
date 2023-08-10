from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

flood_comm_df = pd.read_csv('311_data/wib_calls_311_comm.csv')
flood_comm_stack_df = pd.DataFrame(flood_comm_df.stack()).reset_index()
flood_comm_stack_df = flood_comm_stack_df.rename(columns={'level_0':'Date','level_1':'Community Area',0:'Count Calls'})
flood_comm_totals = pd.DataFrame(flood_comm_stack_df.groupby(['Community Area'])['Count Calls'].sum()).reset_index()
flood_comm_totals.head()

parcel_comm_df = pd.read_csv('parcel_data/res_parcel_stats_by_comm.csv')
parcel_comm_df = parcel_comm_df.rename(columns={'CommunityArea': 'Community Area'})
parcel_comm_df = parcel_comm_df[['Community Area', 'MeanBldgAge', 'ParcelCount', 'MeanBldgValue']]
parcel_comm_df.head()

flood_parcel_df = flood_comm_totals.merge(parcel_comm_df, on='Community Area')
flood_parcel_df['Count Calls'] = flood_parcel_df['Count Calls'].astype(int)
flood_parcel_df.head()

# The Loop, Near North Side, and Lincoln Park are outliers, so removing them
flood_parcel_sub = flood_parcel_df.loc[~flood_parcel_df['Community Area'].isin(['LOOP', 'NEAR NORTH SIDE', 'LINCOLN PARK'])].copy()
flood_parcel_sub.plot(title='Flooding Calls v Mean Parcel Value', kind='scatter', x='MeanBldgValue', y='Count Calls')

flood_parcel_sub.corr()




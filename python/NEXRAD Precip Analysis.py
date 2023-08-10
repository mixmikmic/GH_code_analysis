import numpy as np
import pandas as pd
import os, re
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

nex_df = pd.read_csv('data/nexrad_testing_10k.csv')
nex_df['timestamp'] = pd.to_datetime(nex_df['timestamp'])
nex_df = nex_df.set_index(pd.DatetimeIndex(nex_df['timestamp']))
nex_df = nex_df.dropna()
print(nex_df.dtypes)
nex_df.head()

zip_cols = nex_df.columns.values[1:len(nex_df.columns.values)-1]
zip_precip = nex_df[zip_cols].sum()
zip_precip = zip_precip.sort_values(ascending=False)
zip_precip.plot(kind='bar')

zip_area = pd.read_csv('data/zip_code_area.csv')
print(zip_area.dtypes)
zip_area.head()

zip_precip.head()

zip_precip_sum = pd.DataFrame(zip_precip).reset_index()
zip_precip_sum = zip_precip_sum.rename(columns={'index':'zip',0:'precip'})
zip_precip_sum.head()

zip_precip_sum['zip'] = zip_precip_sum['zip'].astype(int)
zip_precip_area = zip_precip_sum.merge(zip_area, on='zip')
zip_precip_area.head()

zip_precip_area.plot(x='precip',y='shape_area')




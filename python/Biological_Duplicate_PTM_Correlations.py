import bio_duplicate_correlation

fig_data = bio_duplicate_correlation.compare_duplicate_non_duplicate_correlation()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.style.use('ggplot')

fig = fig_data.plot(kind='bar', figsize=(10,5))
full_title = 'PTM Correlation of Bio-Replicates and Non-Bio-Replicates'
fig.set_title(full_title)

import bio_duplicate_correlation
df_scatter = bio_duplicate_correlation.view_scatter()

df_scatter.shape
cols = df_scatter.columns.tolist()

import pandas as pd
import numpy as np
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

# s.plot(kind='bar', figsize=(10,5))
# df_scatter = df_scatter.transpose()
df_scatter.plot(kind='scatter', figsize=(10,5), x=cols[4], y=cols[5])




df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])

df.plot(kind='scatter', x='a', y='b')

inst_series = df_scatter[cols[0]]

print(type(inst_series))

inst_series.hist(bins=100)




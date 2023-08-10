from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

flood_comm_df = pd.read_csv('311_data/flood_calls_311_comm.csv')
flood_comm_df.head()

flood_comm_df['Created Date'] = pd.to_datetime(flood_comm_df['Created Date'])
flood_comm_df = flood_comm_df.set_index(pd.DatetimeIndex(flood_comm_df['Created Date']))
flood_comm_df = flood_comm_df[flood_comm_df.columns.values[1:]]
flood_comm_df.head()

# Get the sum for each day, plot over time
flood_comm_sum = flood_comm_df.sum(axis=1)
flood_comm_sum.head()

plt.rcParams["figure.figsize"] = [15, 5]
flood_comm_sum.plot()

wib_comm_df = pd.read_csv('311_data/wib_calls_311_comm.csv')
wib_comm_df['Created Date'] = pd.to_datetime(wib_comm_df['Created Date'])
wib_comm_df = wib_comm_df.set_index(pd.DatetimeIndex(wib_comm_df['Created Date']))
wib_comm_df = wib_comm_df[wib_comm_df.columns.values[1:]]
wib_comm_sum = wib_comm_df.sum(axis=1)
wib_comm_sum.head()

wos_comm_df = pd.read_csv('311_data/wos_calls_311_comm.csv')
wos_comm_df['Created Date'] = pd.to_datetime(wos_comm_df['Created Date'])
wos_comm_df = wos_comm_df.set_index(pd.DatetimeIndex(wos_comm_df['Created Date']))
wos_comm_df = wos_comm_df[wos_comm_df.columns.values[1:]]
wos_comm_sum = wos_comm_df.sum(axis=1)
wos_comm_sum.head()

combined_calls = pd.DataFrame()
combined_calls['Water in Basement'] = wib_comm_sum
combined_calls['Water in Street'] = wos_comm_sum
combined_calls.plot()




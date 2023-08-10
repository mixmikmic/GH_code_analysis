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

flood_comm_stack = flood_comm_df.stack()
flood_comm_stack.head()

flood_comm_stack_df = pd.DataFrame(flood_comm_stack).reset_index()
flood_comm_stack_df.head()

flood_comm_stack_df = flood_comm_stack_df.rename(columns={'level_0':'Date','level_1':'Community Area',0:'Count Calls'})
flood_comm_totals = flood_comm_stack_df.groupby(['Community Area'])['Count Calls'].sum()
flood_comm_totals.head()

flood_comm_sum = flood_comm_totals.reset_index()
flood_comm_sum.head()

flood_comm_top = flood_comm_sum.sort_values(by='Count Calls', ascending=False)[:20]
flood_comm_top.plot(kind='bar',x='Community Area',y='Count Calls')

# WBEZ zip data
wbez_zip = pd.read_csv('wbez_flood_311_zip.csv')
wbez_zip_top = wbez_zip.sort_values(by='number_of_311_calls',ascending=False)[:20]
wbez_zip_top.plot(kind='bar',x='zip_code',y='number_of_311_calls')

flood_zip_df = pd.read_csv('311_data/flood_calls_311_zip.csv')
flood_zip_df['Created Date'] = pd.to_datetime(flood_zip_df['Created Date'])
flood_zip_df = flood_zip_df.set_index(pd.DatetimeIndex(flood_zip_df['Created Date']))
# Setting to same time frame as WBEZ
flood_zip_df = flood_zip_df['2009-01-01':'2015-12-30']
flood_zip_df = flood_zip_df[flood_zip_df.columns.values[1:]]
flood_zip_stack = pd.DataFrame(flood_zip_df.stack()).reset_index()
flood_zip_stack = flood_zip_stack.rename(columns={'level_0':'Created Date','level_1':'Zip Code',0:'Count Calls'})
flood_zip_sum = flood_zip_stack.groupby(['Zip Code'])['Count Calls'].sum()
flood_zip_sum = flood_zip_sum.reset_index()
flood_zip_sum.head()

flood_zip_top = flood_zip_sum.sort_values(by='Count Calls', ascending=False)[:20]
flood_zip_top.plot(kind='bar',x='Zip Code',y='Count Calls')

fig, axs = plt.subplots(1,2)
plt.rcParams["figure.figsize"] = [15, 5]
wbez_zip_top.plot(title='WBEZ Data', ax=axs[0], kind='bar',x='zip_code',y='number_of_311_calls')
flood_zip_top.plot(title='FOIA Data', ax=axs[1], kind='bar',x='Zip Code',y='Count Calls')

flood_comm_time = flood_comm_stack_df.copy()
flood_comm_time['Date'] = pd.to_datetime(flood_comm_time['Date'])
flood_comm_time = flood_comm_time.set_index(pd.DatetimeIndex(flood_comm_time['Date']))
flood_comm_time = flood_comm_time['2009-01-01':'2015-12-30']
flood_comm_time_sum = flood_comm_time.groupby(['Community Area'])['Count Calls'].sum()
flood_comm_time_sum = flood_comm_time_sum.reset_index()
flood_comm_time_sum.head()

flood_comm_time_top = flood_comm_time_sum.sort_values(by='Count Calls', ascending=False)[:20]
flood_comm_time_top.plot(kind='bar',x='Community Area',y='Count Calls')




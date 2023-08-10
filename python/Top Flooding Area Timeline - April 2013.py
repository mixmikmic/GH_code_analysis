from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

comm_df = pd.read_csv('311_data/flood_calls_311_comm.csv')
# Taking top 6 community areas for flooding 
comm_df = comm_df[['Created Date','AUSTIN','ROSELAND','CHICAGO LAWN','AUBURN GRESHAM','WASHINGTON HEIGHTS','CHATHAM']].copy()
comm_df['Created Date'] = pd.to_datetime(comm_df['Created Date'])
comm_df = comm_df.set_index(pd.DatetimeIndex(comm_df['Created Date']))
# Just get calls since the beginning of 2010
comm_df = comm_df['2010-01-01':]
comm_df.head()

comm_month_grp = comm_df.groupby([comm_df.index.year,comm_df.index.month]).sum()
comm_month_grp.head()

comm_month_grp.sum().plot(title='Top 6 Community Areas by Flooding 311 Calls - 2010-Present', kind='bar')

plt.rcParams["figure.figsize"] = [15, 5]
comm_month_grp.plot(title='Flooding 311 Calls by Top 6 Community Areas - 2010-Present')

april_df = comm_df['2013-04-01':'2013-05-01']
april_df.plot(title='Flooding 311 Calls by Top 6 Community Areas - April 2013')

april_df.sum().plot(title='Flooding 311 Calls - April 2013', kind='bar')

mid_april_df = comm_df['2013-04-15':'2013-04-25']
mid_april_df.plot(title='Flooding 311 Calls by Top 6 Community Areas - Mid-April 2013')




from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

wib_comm_df = pd.read_csv('311_data/wib_calls_311_comm.csv')
wos_comm_df = pd.read_csv('311_data/wos_calls_311_comm.csv')
wib_comm_df.head()

wib_comm_stack = wib_comm_df[wib_comm_df.columns.values[1:]].stack().reset_index()
wos_comm_stack = wos_comm_df[wos_comm_df.columns.values[1:]].stack().reset_index()
wib_comm_stack.head()

wib_comm_grp = pd.DataFrame(wib_comm_stack.groupby(['level_1'])[0].sum()).reset_index()
wib_comm_grp = wib_comm_grp.rename(columns={'level_1':'Community Area', 0: 'Basement Calls'})
wos_comm_grp = pd.DataFrame(wos_comm_stack.groupby(['level_1'])[0].sum()).reset_index()
wos_comm_grp = wos_comm_grp.rename(columns={'level_1':'Community Area', 0: 'Street Calls'})
comm_grp_merge = wib_comm_grp.merge(wos_comm_grp, on='Community Area')
comm_grp_merge.head()

## Making basement to street call ratio column
comm_grp_merge['Basement-Street Ratio'] = comm_grp_merge['Basement Calls'] / comm_grp_merge['Street Calls']
comm_grp_merge.head()

comm_more_basement = comm_grp_merge.sort_values(by='Basement-Street Ratio', ascending=False)[:15]
comm_more_street = comm_grp_merge.sort_values(by='Basement-Street Ratio')[:15]
comm_more_basement.head()

fig, axs = plt.subplots(1,2)
plt.rcParams["figure.figsize"] = [15, 5]
comm_more_basement.plot(title='More Basement Flooding Areas', ax=axs[0], kind='bar',x='Community Area',y='Basement-Street Ratio')
comm_more_street.plot(title='More Street Flooding Areas', ax=axs[1], kind='bar',x='Community Area',y='Basement-Street Ratio')




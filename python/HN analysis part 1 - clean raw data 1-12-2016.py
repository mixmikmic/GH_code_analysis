import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

ls

HN_df = pd.read_csv('HN_metrics_28_11.csv')

HN_df.head()

HN_df['structure'].unique()

HN_df['patID'].unique()

HN_df['Case'] = HN_df['patID'].str.split('_').str.get(0) # get Case2 etc 
HN_df['algo'] = HN_df['patID'].str.split('_').str.get(1) # get AXB or Dm

HN_df.tail()

HN_df['algo'].replace(to_replace='AXB', value='Dm', inplace=True)# replace

HN_df['algo'].describe()

HN_df['algo'].count()/2

HN_df.head()

HN_df[(HN_df['algo']=='AAA') & (HN_df['metric'] == 'D50%')].groupby(['structure']).count()['observed']  # these are my structure counts

HN_df['structure'].replace(to_replace='CTV 54Gy', value='CTV54', inplace=True)# replace
HN_df['structure'].replace(to_replace='CTV 65Gy', value='CTV65', inplace=True)# replace
HN_df['structure'].replace(to_replace='BODY', value='Body', inplace=True)# replace

HN_df[(HN_df['algo']=='AAA') & (HN_df['metric'] == 'D50%')].groupby(['structure']).count()['observed'] 

HN_df[(HN_df['algo']=='Dm') & (HN_df['metric'] == 'D50%')].groupby(['structure']).count()['observed'] 

HN_df[(HN_df['structure']=='GTV')]['patID'].value_counts()  # Case 2 is our problem

HN_df[(HN_df['structure']=='GTV') & (HN_df['Case']=='Case2') & (HN_df['algo']=='AAA')]

structures_to_keep = ['PTV1', 
                      'PTV2', 
                      'CTV65',
                      'Body',
                      'Brain Stem',
                      'L Parotid', 
                      'R Parotid',
                      'Spinal Cord']

HN = HN_df[HN_df['structure'].isin(structures_to_keep)] 

HN.head()

AAA = HN[HN['algo']=='AAA']
Dm = HN[HN['algo']=='Dm']
HN_to_save = pd.merge(AAA, Dm, how='inner', on=['metric', 'Case', 'structure'])  # Get merged set

HN_to_save.head()

HN_to_save.rename(columns={'observed_x': 'AAA'}, inplace=True) 
HN_to_save.rename(columns={'observed_y': 'Dm'}, inplace=True) 

HN_to_save.head()

HN_to_save.to_csv('HN_df_clean_28_11.csv', index=False) 

HN_to_save = HN_to_save.drop(['patID_x','patID_y', 'algo_x', 'algo_y'], 1)  
HN_to_save.head()

HN_to_save = HN_to_save[['Case', 'structure', 'metric', 'AAA', 'Dm']]

HN_to_save.head()  # check against raw, OK

HN_to_save.to_csv('HN_df_clean_28_11.csv', index=False)  # write to file




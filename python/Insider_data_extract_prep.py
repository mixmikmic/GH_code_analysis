import pandas as pd

FU2_users = pd.read_csv('./Data_subset/FU2.csv')

FU2_users.shape

FU2_users.head()

device = pd.read_csv('./CERT_Data/r3.2/device.csv')

device.shape

device.head()

logon = pd.read_csv('./CERT_Data/r3.2/logon.csv')

logon.shape

logon.head()

len(logon['pc'].unique())

files = pd.read_csv('./CERT_Data/r3.2/file.csv')

files.shape

files.head()

psychometric = pd.read_csv('./CERT_Data/r3.2/psychometric.csv')

psychometric.shape

psychometric.head()

# device
fu2_device = device[device.user.isin(FU2_users.user_id)]

fu2_device.shape

fu2_device.head()

# logon
fu2_logon = logon[logon.user.isin(FU2_users.user_id)]

fu2_logon.shape

fu2_logon.head()

# file

fu2_file = files[files.user.isin(FU2_users.user_id)]

fu2_file.shape

fu2_file.head()

# psychometric

fu2_psychometric = psychometric[psychometric.user_id.isin(FU2_users.user_id)]

fu2_psychometric.shape


fu2_device.isnull().sum()

fu2_logon.isnull().sum()

fu2_file.isnull().sum()

fu2_psychometric.isnull().sum()

fu2_device.to_csv('./Data_Subset/fu2_device.csv', index = False)

fu2_logon.to_csv('./Data_Subset/fu2_logon.csv', index = False)

fu2_file.to_csv('./Data_Subset/fu2_file.csv', index = False)

fu2_psychometric.to_csv('./Data_Subset/fu2_psychometric.csv', index = False)

logon1 = fu2_logon.groupby(['user', 'pc', 'activity']).size().reset_index() # reset_index() to return result as a df.

logon1

type(logon1)

# Rename column '0' as 'activity_cnt'

logon1.rename(columns={0:'activity_cnt'}, inplace=True)

logon1.head()

# Consider only Logoff (as edge weight for the Graph)
# Subset records for only Logoff from the above dataframe 'logon1'

logoff = logon1.loc[logon1['activity'] == 'Logoff']

logoff.shape

logoff.head()

# Save the Logoff event data 

logoff.to_csv('./Data_Subset/graph_params_logoff.csv', index = False)

# edges and weights tuple params for the Graph
weighted_edges = [(row['user'], row['pc'], row['activity_cnt']) for idx, row in logoff.iterrows()]

weighted_edges

fu2_logon.groupby(['user', 'pc'])['pc'].count()




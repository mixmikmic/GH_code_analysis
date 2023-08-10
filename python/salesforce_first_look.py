import pandas as pd
get_ipython().magic('matplotlib inline')

get_ipython().run_cell_magic('bash', '', 'ls data/')

df = pd.read_csv('data/FUF Salesforce Data Dump.csv')
df.head()

df.shape

df.describe()

df.filter(regex='.*Status').count()

df[['Survey2mStatus', 'Survey3yrStatus']].dropna().shape

df['Survey2mStatus'].value_counts()

df['Survey3yrStatus'].value_counts()

df['Survey2mStatus_num'] = df['Survey2mStatus'].str.extract('(\d+) \w+', expand=False).astype(float)
df['Survey3yrStatus_num'] = df['Survey3yrStatus'].str.extract('(\d+) \w+', expand=False).astype(float)
df['status_change'] = df['Survey2mStatus_num'] - df['Survey3yrStatus_num']
df['status_change'].value_counts().sort_index()

df.Neighborhood.value_counts()




import pandas as pd

data = pd.read_table('../data/interim/verified-checklist.tsv', dtype=object)

data.head()

data['index'].count()

data.groupby('gbifapi_usageKey').first().reset_index()['index'].count()

data[data['nameMatchValidation'].str.contains('verify', na=False)]['index'].count()

data[data['nameMatchValidation'].str.contains('ok', na=False)].groupby('gbifapi_usageKey').first().reset_index()['index'].count()




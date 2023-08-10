import json
import urllib.parse

import numpy as np
import pandas as pd
import requests

get_ipython().magic('matplotlib inline')

# It might be overkill, but I figured it best
# for legibility to separate query arguments as a dict

params_dict = {
    "q":"projectid:30072",
    "per_page":"1000"
}

params_encoded = urllib.parse.urlencode(params_dict)

r = requests.get('https://www.documentcloud.org/api/search.json', params=params_encoded)

r.json()['documents'][0]

documentcloud_df = pd.read_json(json.dumps(r.json()['documents']))

documentcloud_df.head()

df = pd.read_csv('armories_data - 20161201.csv',dtype={'Oregonian ID':'str','Inspection year':'str'})

df.ix[975]

df.head()

df_modified = df.replace(to_replace=['Unknown','Yes','No'],value=[np.nan,1,0])

df_modified.head()

df_states = df_modified[['State','Oregonian ID','Lead present?','Had firing range?']].groupby('State').sum().reset_index()

df_state_count = df_modified[['State','Oregonian ID']].groupby('State').count().reset_index()

df_state_count.rename(columns={'Oregonian ID':'Armory Count'},inplace=True)

df_state_values = df_modified[['State','Inspection conducted?','Inspection report available?','Lead present?','Lead present outside range?','Had firing range?']].groupby('State').sum().reset_index()

df_states = pd.merge(df_state_count,df_state_values)

df_state_values['Rate of lead in state'] = round(df_states['Lead present?']/df_states['Armory Count'],2)

df_state_values['Rate of inspection'] = round(df_states['Inspection conducted?']/df_states['Armory Count'],2)

df_state_values.sort_values(by=['Rate of inspection'],ascending=False)

df_state_values[['Rate of inspection']].sort_values(
    by='Rate of inspection', ascending=False).plot(
    kind='bar',
    title='Rate of armory inspection by state',
    legend=False,
    x=df_state_values['State'])


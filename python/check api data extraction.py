import io
import requests
import json
import http.client

import pandas as pd

import altair as alt
#magic

get_ipython().run_line_magic('matplotlib', 'inline')

conn = http.client.HTTPSConnection("api.resourcewatch.org")

conn.request("GET", "/v1/query/8027f9dc-8531-46d7-bd3c-e48393c14dc3?sql=SELECT%20*%20FROM%20index_8027f9dc853146d7bd3ce48393c14dc3")

res = conn.getresponse()
data = res.read()

url = r"http://api.resourcewatch.org/v1/query/8027f9dc-8531-46d7-bd3c-e48393c14dc3?sql=SELECT%20*%20FROM%20index_8027f9dc853146d7bd3ce48393c14dc3"
response = requests.get(url)

js = response.json()

df = pd.DataFrame(js["data"])

df.head()

year_cols =[col for col in df.columns.tolist() if col.isdigit()]

id_cols = set(df.columns.tolist()) - set(year_cols)

df_melt = df.melt(id_vars = id_cols,
                  value_vars = year_cols, 
                  var_name = "year",
                  value_name = "mortality"
                 )

df_melt.head()

df_melt.describe()

df_melt.info()

countries = pd.Series(df_melt["Country_Name"].unique()).sample(frac = 0.1, 
                                                                random_state = 42
                                                               ).tolist()
    
mask = (df_melt["Country_Name"].isin(countries)) & (df_melt["mortality"] > 0)

c = alt.Chart(df_melt[mask]).mark_line().encode( x = 'year',
                                          y='mortality',
                                          color='Country_Name'
                                         )
c


import numpy as np
import pandas as pd
import datetime
import urllib

query = ("https://data.phila.gov/resource/4t9v-rppq.json?$where=requested_datetime%20between%20%272016-09-01T00:00:00%27%20and%20%272016-10-01T00:00:00%27")
df = pd.read_json(query, convert_dates=['expected_datetime','requested_datetime','updated_datetime'])

df.to_pickle('311_Requests.pickle')
df.to_csv('311_Requests.csv')


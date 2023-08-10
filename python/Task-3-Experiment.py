import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from pymongo import MongoClient
get_ipython().magic('matplotlib inline')

client = MongoClient('mongodb')
db = client.dp
collection = db.divorce

data = db.divorce.find()[0]['data']
for entry in data:
    entry['DIVORCES'] = entry['values'][0]['NUMBER']
    s = entry['DURATION']
    tmp = re.findall(r'\d+', s)
    if (len(tmp) == 1):
        tmp[0] = 0
    del entry['values']
    del entry['NUTS1']
    del entry['NUTS2']
    entry['DURATION'] = tmp[0]

data_json = json.dumps(data)

df = pd.read_json(data_json)
filtered = df[df.DURATION == 10].filter(items=['DIVORCES','REF_YEAR'])
filtered

filtered.plot.bar(x='REF_YEAR',y='DIVORCES')




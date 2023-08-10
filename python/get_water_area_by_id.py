import hydroengine as he

import json
from shapely.geometry import mapping, shape

import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

sns.set(color_codes=True)

# Ethiopia
geometry = { "type": "Point", "coordinates": [37.375, 11.572] }
id_only = True
lake_ids = he.get_lakes(geometry, id_only)

print(lake_ids)

lake_id = lake_ids[0]

ts = he.get_lake_time_series(lake_id, 'water_area')

d = pd.DataFrame(ts)
d['time'] = pd.to_datetime(d['time'], unit='ms')
plt.plot(d.time, d.water_area, '.')

lake = he.get_lake_by_id(lake_id)

s = shape(lake['geometry'])
s

lake['properties']


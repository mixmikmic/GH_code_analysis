import hydroengine as he

import json
from shapely.geometry import mapping, shape

import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

sns.set(color_codes=True)

# Limpopo downstream
geometry = { "type": "Point", "coordinates": [33.56529235839844, -24.895779429137615] }
lakes = he.get_lakes(geometry)

len(lakes['features'])

lakes['features'][0]

lake_id = 183160

ts = he.get_lake_time_series(lake_id, 'water_area')

d = pd.DataFrame(ts)

d['time'] = pd.to_datetime(d['time'], unit='ms')

plt.plot(d.time, d.water_area, '.')

# remove empty values, the algorithm should be actually smarter, distinguishing between NODATA and NOT WATER
d = d[d.water_area > 0]

plt.plot(d.time, d.water_area, '-')

lake = he.get_lake_by_id(lake_id)

s = shape(lake['geometry'])
s

lake['properties']


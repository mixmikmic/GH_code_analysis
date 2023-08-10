import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

locations_csv = StringIO("""timestamp,mmsi,lon,lat,sog,cog,heading
2018-03-19 07:16:03.114,266232000,24.28642667,65.184065,13.7,92.8,92
2018-03-19 07:17:03.202,266232000,24.29567667,65.18406833,14.2,88.1,89
2018-03-19 07:18:02.787,266232000,24.30479,65.18403167,14,90.8,90
2018-03-19 07:18:56.815,266232000,24.31323667,65.18398167,14.1,91.4,91
2018-03-19 07:19:50.959,266232000,24.32111333,65.18374667,13.6,93,93
2018-03-19 07:21:38.54,266232000,24.337,65.18353,13.1,93,93""")

df = pd.read_csv(locations_csv, sep=",")

df

plt.scatter('lon', 'lat', data=df)




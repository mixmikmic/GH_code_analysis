import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import welly
from welly import Well
import lasio
import glob
from sklearn import neighbors
import pickle
welly.__version__

from ipyleaflet import *

get_ipython().run_cell_magic('timeit', '', 'import os\nenv = %env')

pd.set_option('display.max_rows', 2000)

picks_dic = pd.read_csv('../../SPE_006_originalData/OilSandsDB/PICKS_DIC.TXT',delimiter='\t')
picks = pd.read_csv('../../SPE_006_originalData/OilSandsDB/PICKS.TXT',delimiter='\t')
wells = pd.read_csv('../../SPE_006_originalData/OilSandsDB/WELLS.TXT',delimiter='\t')
gis = pd.read_csv('../../well_lat_lng.csv')
picks_new=picks[picks['HorID']==13000]
picks_paleoz=picks[picks['HorID']==14000]
df_new = pd.merge(wells, picks_new, on='SitID')
df_paleoz = pd.merge(wells, picks_paleoz, on='SitID')
#### NOTE: This now includes the GIS or well_lat_lng dataset too!
df_gis = pd.merge(df_paleoz, gis, on='SitID')
df_new=pd.merge(df_gis, df_new, on='SitID')
df_new.head()

position = df_new[['lat','lng']]

position

center = [54.840471, -110.269399]
zoom = 6

m = Map(default_tiles=TileLayer(opacity=1.0), center=center, zoom=zoom)
m.interact(zoom=(5,10,1))

mark = Marker(location=[54.873181, -110.269399])
mark.visible
m += mark


m


print(position[0:][0:1]['lat'])
type(position[0:][0:1]['lat'])

print(position[0:][0:1]['lat'][0])
type(position[0:][0:1]['lat'][0])

position

position[0:]

position[1:][0:1]

len(position[0:])

print(position[0:][0:1]['lat'][0])

print(position[0:][0:1])

new_position = position.values.tolist()
print(new_position)

df_all_columns = df_new.values.tolist()

df_new

#### making a sub data frame for the map
#### Pick_x is base McMurray I think and Pick is top McMurray
df_for_map = df_new[['lat','lng','UWI','Pick','Quality','Pick','Pick_x']]
df_for_map.head()

circle = []
for row in new_position:
    c = Circle(location=list(row)[0:2], radius=1000)
    print(c)
    circles.append(c)
    m.add_layer(c)

circle = []
for row in new_position:
    c = Circle(location=list(row), radius=1000)
    print(c)
    circles.append(c)
    m.add_layer(c)




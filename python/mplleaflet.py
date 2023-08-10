import os

import geopandas as gpd
import matplotlib.pyplot as plt

import mplleaflet

get_ipython().magic('matplotlib inline')

df = gpd.read_file('data/geo_export_bbd3984d-4213-4949-8df3-130185e1a6df.shp')

ax = df.plot(figsize=(15,15))

# The display call inserts the html in the IPython notebook.
# An interactive, panable, zoomable map.
mplleaflet.display(fig=ax.figure, crs=df.crs)




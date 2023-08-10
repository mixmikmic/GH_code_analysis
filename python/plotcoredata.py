get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import Image
from IPython.display import HTML

Image(filename='Chillrud-Abstract.png',width=300)

Image(filename='central_park_core.jpg',width=600)

Image(filename='Seawolf.jpg',width=400)

Image(filename='127_2800.jpg',width=300)

Image(filename='128_2801.jpg',width=300)

Image(filename='EM09-GC-01_comp.jpg',width=150)

import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
from cartopy.io.img_tiles import OSM
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.figure(figsize=(10, 9))

ll_lat=40.7
ll_long=-74.05
ur_lat=40.95
ur_long=-73.8

# Use the tile's projection for the underlying map.
ax = plt.axes(projection=ccrs.PlateCarree())

# Specify a region of interest
ax.set_extent([ll_long, ur_long, ll_lat, ur_lat],    
              ccrs.PlateCarree())                     # this is x1,x2, y1,y2

gg_tiles = GoogleTiles(style='street') #we can change the style of teh figure from 'streets' to 'terrain' to 'satellite'
ax.add_image(gg_tiles, 11,alpha=1)   #the number is the zoom level and the alpha is if it is transparent

ax.scatter(-73.94,40.87766667, color='red', s=100, transform=ccrs.PlateCarree())
ax.set_title('Approximate Core Location in Red"')




import geopandas as gpd
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


import os
os.getcwd()

fp = 'C:\\Users\\Shakur\\Documents\\scripts\\Projects_DEC17\\DATA\\Europe_borders.shp'

data = gpd.read_file(fp)

data.head()

data.plot()

data.crs

data['geometry'].head()

data_proj = data.copy()

data_proj = data_proj.to_crs(epsg=3035)

data_proj['geometry'].head()

import matplotlib.pyplot as plt

data.plot(facecolor='gray')
plt.title("WGS84 Projection")
plt.tight_layout()

data_proj.plot(facecolor='blue')
plt.title('ETRS Lamber Azimuthal Equal Area Project')

plt.tight_layout()

from fiona.crs import from_epsg

data_proj.crs = from_epsg(3035)

data_proj.crs

outfp = 'C:\\Users\\Shakur\\Documents\\scripts\\Projects_DEC17\\DATA\\Europe_borders_epsg3035.shp'

data_proj.to_file(outfp)




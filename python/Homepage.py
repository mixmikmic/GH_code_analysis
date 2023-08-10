import holoviews as hv
import holocube as hc
from cartopy import crs
from cartopy import feature as cf

hv.notebook_extension()

get_ipython().run_cell_magic('opts', 'GeoFeature [projection=crs.Geostationary()]', '\ncoasts  = hc.GeoFeature(cf.COASTLINE)\nborders = hc.GeoFeature(cf.BORDERS)\nocean   = hc.GeoFeature(cf.OCEAN)\n\nocean + borders + (ocean*borders).relabel("Overlay")')

import iris
surface_temp = iris.load_cube(iris.sample_data_path('GloSea4', 'ensemble_001.pp'))
print surface_temp.summary()

get_ipython().run_cell_magic('opts', "GeoImage [colorbar=True] (cmap='viridis')", "(hc.HoloCube(surface_temp).groupby(['time'], group_type=hc.Image) * hc.GeoFeature(cf.COASTLINE))")


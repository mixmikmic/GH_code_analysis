get_ipython().run_cell_magic('HTML', '', '<style>\n.container{width:75% !important;}\n.text_cell_rendered_html{width:20% !important;}\n</style>')

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6



import xarray as xr
import numpy as np
import pandas as pd
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf

import cartopy
from cartopy import crs as ccrs

from bokeh.tile_providers import STAMEN_TONER
from bokeh.models import WMTSTileSource

hv.notebook_extension('bokeh')


from bokeh.models import (
    GeoJSONDataSource,
    HoverTool,
    LinearColorMapper
)
from bokeh.plotting import figure
from bokeh.palettes import Viridis6
from bokeh.io import output_notebook, show
output_notebook()

data = pd.read_csv("data/listings/08042017/listings.csv", low_memory=False)
num_hogares = pd.read_csv("data/num_hogares/NumHogaresYFamilias2011.csv", sep=";", thousands='.')

num_hogares.head()

num_hogares.columns = num_hogares.columns.str.lstrip() #get rid of trailing spaces

def drop_digits(in_str): #sorry for this... I'm just too lazy sometimes
    digit_list = "1234567890"
    for char in digit_list:
        in_str = in_str.str.replace(char, "")

    return in_str

num_hogares.Barrio = drop_digits(num_hogares.Barrio)
num_hogares.Barrio = num_hogares.Barrio.str.replace(".", "")

#count the number of listings for each neighbourhood
n_airbnbs_barri= data.neighbourhood_cleansed.value_counts()

#make shure that Series contains neighbourhoods that are in census
n_airbnbs_barri = n_airbnbs_barri[n_airbnbs_barri.index.isin(num_hogares.Barrio)] 

num_hogares = num_hogares[num_hogares.Barrio.isin(n_airbnbs_barri.index)]

num_hogares.index = num_hogares.Barrio
num_hogares.drop("Barrio", axis=1)[:2]

num_hogares = num_hogares.NumHogares
#num_hogares.drop(["Can Peguera", "Baró de Viver", "Torre Baró", "Vallbona"], inplace=True)#outliers

#calculate the density
density = n_airbnbs_barri/num_hogares

n_airbnbs = pd.DataFrame(n_airbnbs_barri)
n_households = pd.DataFrame(num_hogares)

#Add total number of airbnbs and households to df
n_airbnbs['N_Barri'] =  n_airbnbs.index
n_households['N_Barri']= n_households.index

density = pd.DataFrame({"N_Barri":density.index, "value":density.values})

density = density.merge(n_airbnbs, how='left', on='N_Barri')
density = density.merge(n_households, how='left', on='N_Barri')

density.columns = ['N_Barri','value','n_airbnb','n_households']

density.head()

barri_json_path = r"data/divisiones_administrativas/barris/barris_geo.json"
with open(barri_json_path, 'r') as f:
    geo_source = GeoJSONDataSource(geojson=f.read())


TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

p = figure(title="Neighbourhoods", tools=TOOLS, x_axis_location=None,
           y_axis_location=None, width=800, height=800)

p.grid.grid_line_color = None

p.patches('xs', 'ys', fill_alpha=0.7, 
          line_color='white', line_width=1, source=geo_source)

hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [("Neighbourhood", "@N_Barri")]

show(p)

shapefile = "data/divisiones_administrativas/barris/shape/barris_geo.shp"
shapes = cartopy.io.shapereader.Reader(shapefile)


density_hv = hv.Dataset(density)

density_hv.data.dropna(inplace=True)

get_ipython().run_cell_magic('opts', 'Overlay [width=1000 height=1000 xaxis=None yaxis=None] ', '%%output filename="holoviewsmap"\n\ngv.Shape.from_records(shapes.records(), density_hv, on=\'N_Barri\', value=\'value\',\n                      index=[\'N_Barri\',\'n_airbnb\',\'n_households\'], #hack to make them appear at the hoovertool\n                      crs=ccrs.PlateCarree(), group="Densitat Airbnb Barcelona per nombre d\'hogars",\n                      drop_missing=False)\n\n%%opts Shape (cmap=\'Reds\') [tools=[\'hover\'] width=1000 height=1000 colorbar=True toolbar=\'above\' xaxis=None yaxis=None]')






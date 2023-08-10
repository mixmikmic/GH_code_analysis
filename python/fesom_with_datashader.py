import sys
sys.path.append("../")

import pyfesom as pf
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
#%matplotlib notebook
from matplotlib import cm

import pandas as pd

import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf
from datashader.bokeh_ext import InteractiveImage
import bokeh.plotting as bp
from datashader.utils import export_image
from functools import partial
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from IPython.core.display import HTML, display
from bokeh.models import LogColorMapper, LogTicker, ColorBar, LinearColorMapper, BasicTicker
#from datashader.bokeh_ext import create_ramp_legend, create_categorical_legend
from bokeh.palettes import Viridis256
from matplotlib import cm
from bokeh.plotting import figure, output_notebook, show
Viridis256.reverse()
output_notebook()

meshpath  ='../../../../FESOM/mesh/'
mesh = pf.load_mesh(meshpath, get3d=True)

mesh

df = pd.DataFrame({'lon':mesh.x2, 'lat':mesh.y2, 'topo':mesh.topo})

def create_image(x_range=(-180,180), y_range=(-90,90), w=120500, h=500):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range,y_range=y_range )
    agg = cvs.points(df, 'lon', 'lat', ds.mean('topo'))
    ccm = partial(colormap_select, reverse=(background!="black"))
    img = tf.shade(agg, cmap=Viridis256, how='eq_hist')
    dd = tf.dynspread(img, threshold=0.5, max_px=4, shape='circle')
    
    return dd


bp.output_notebook()

background = "black"
export = partial(export_image, export_path="export", background=background)


def base_plot(tools='pan,wheel_zoom,reset, save', webgl=False):
    p = bp.figure(tools=tools, plot_width=700, plot_height=600,
        x_range=(-180,180), y_range=(-90,90), outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0)   
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=5000)
    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))
    p.add_layout(color_bar, 'right')
    return p
# export(create_image(*NYC),"NYCT_hot")
p = base_plot()

#p.add_tile(STAMEN_TERRAIN)
#url="http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.png"
#url="http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png"
#tile_renderer = p.add_tile(WMTSTileSource(url=url))
#tile_renderer.alpha=1.0 if background == "black" else 0.15
#export(create_image,"NYCT_hot")
InteractiveImage(p, create_image)




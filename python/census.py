import datashader as ds
import datashader.transfer_functions as tf
import dask.dataframe as dd
import numpy as np

get_ipython().run_cell_magic('time', '', "df = dd.from_castra('data/census.castra')\ndf = df.cache(cache=dict)")

df.tail()

USA =          ((-13884029,  -7453304), (2698291, 6455972))
LakeMichigan = ((-10206131,  -9348029), (4975642, 5477059))
Chicago =      (( -9828281,  -9717659), (5096658, 5161298))
Chinatown =    (( -9759210,  -9754583), (5137122, 5139825))

NewYorkCity =  (( -8280656,  -8175066), (4940514, 4998954))
LosAngeles =   ((-13195052, -13114944), (3979242, 4023720))
Houston =      ((-10692703, -10539441), (3432521, 3517616))
Austin =       ((-10898752, -10855820), (3525750, 3550837))
NewOrleans =   ((-10059963, -10006348), (3480787, 3510555))
Atlanta =      (( -9448349,  -9354773), (3955797, 4007753))

x_range,y_range = USA

plot_width  = int(1000)
plot_height = int(plot_width*7.0/12)

black_background = True

from IPython.core.display import HTML, display
display(HTML("<style>.container { width:100% !important; }</style>"))

def export(img,filename,fmt=".png",_return=True):
    """Given a datashader Image object, saves it to a disk file in the requested format"""
    if black_background: 
        img=tf.set_background(img,"black")
    img.to_pil().save(filename+fmt)
    return img if _return else None

def cm(base_colormap, start=0, end=1.0, reverse=not black_background):
    """
    Given a colormap in the form of a list, such as a Bokeh palette,
    return a version of the colormap reversed if requested, and selecting
    a subset (on a scale 0,1.0) of the elements in the colormap list.
    
    For instance:
    
    >>> cmap = ["#000000", "#969696", "#d9d9d9", "#ffffff"]
    >>> cm(cmap,reverse=True)
    ['#ffffff', '#d9d9d9', '#969696', '#000000']
    >>> cm(cmap,0.3,reverse=True)
    ['#d9d9d9', '#969696', '#000000']
    """
    full = reversed(base_colormap) if reverse else base_colormap
    num = len(full)
    return full[int(start*num):int(end*num)]

from datashader.colors import Greys9, Hot, viridis, inferno

get_ipython().run_cell_magic('time', '', "cvs = ds.Canvas(plot_width, plot_height, *USA)\nagg = cvs.points(df, 'meterswest', 'metersnorth')")

export(tf.interpolate(agg, cmap = cm(Greys9), how='linear'),"census_gray_linear")

export(tf.interpolate(agg, cmap = cm(Greys9,0.25), how='linear'),"census_gray_linear")

export(tf.interpolate(agg, cmap = cm(Greys9,0.2), how='log'),"census_gray_log")

export(tf.interpolate(agg, cmap = cm(Greys9,0.2), how='eq_hist'),"census_gray_eq_hist")

print(cm(Hot,0.2))
export(tf.interpolate(agg, cmap = cm(Hot,0.2), how='eq_hist'),"census_ds_hot_eq_hist")

from bokeh.palettes import PuRd9
export(tf.interpolate(agg, cmap=cm(PuRd9), how='eq_hist'),"census_inferno_eq_hist")

export(tf.interpolate(agg, cmap=cm(viridis), how='eq_hist'),"census_viridis_eq_hist")

grays2 = cm([(i,i,i) for i in np.linspace(0,255,99)])
grays2 += ["red"]
export(tf.interpolate(agg, cmap = grays2, how='eq_hist'),"census_gray_redhot1_eq_hist")

if black_background:
      color_key = {'w':'aqua', 'b':'lime',  'a':'red', 'h':'fuchsia', 'o':'yellow' }
else: color_key = {'w':'blue', 'b':'green', 'a':'red', 'h':'orange',  'o':'saddlebrown'}

def create_image(x_range, y_range, w=plot_width, h=plot_height):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'meterswest', 'metersnorth', ds.count_cat('race'))
    img = tf.colorize(agg, color_key, how='eq_hist')
    return img

export(create_image(*USA),"Zoom 0 - USA")

export(create_image(*LakeMichigan),"Zoom 1 - Lake Michigan")

export(create_image(*Chicago),"Zoom 2 - Chicago")

export(tf.spread(create_image(*Chinatown),px=plot_width/400),"Zoom 3 - Chinatown")

export(create_image(*NewYorkCity),"NYC")

export(create_image(*LosAngeles),"LosAngeles")

export(create_image(*Houston),"Houston")

export(create_image(*Atlanta),"Atlanta")

export(create_image(*NewOrleans),"NewOrleans")

export(create_image(*Austin),"Austin")

cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
agg = cvs.points(df, 'meterswest', 'metersnorth', ds.count_cat('race'))

export(tf.interpolate(agg.sel(race='b'), cmap=cm(Greys9,0.25), how='eq_hist'),"USA blacks")

agg2 = agg.where((agg.sel(race=['w', 'b', 'a', 'h']) > 0).all(dim='race')).fillna(0)
export(tf.colorize(agg2, color_key, how='eq_hist'),"USA all")

export(tf.colorize(agg.where(agg.sel(race='w') < agg.sel(race='b')).fillna(0), color_key, how='eq_hist'),"more_blacks")

import bokeh.plotting as bp
from bokeh.models.tiles import WMTSTileSource

bp.output_notebook()

def base_plot(tools='pan,wheel_zoom,reset',webgl=False):
    p = bp.figure(tools=tools, 
        plot_width=int(900*1.5), plot_height=int(500*1.5),
        x_range=x_range, y_range=y_range, outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0, webgl=webgl)
    
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.responsive = True
    
    return p

from datashader.bokeh_ext import InteractiveImage

def image_callback(x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'meterswest', 'metersnorth', ds.count_cat('race'))
    img = tf.colorize(agg, color_key, 'log')
    return tf.dynspread(img,threshold=0.75, max_px=8)

p = base_plot()

url="http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.png"
#url="http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png"
tile_renderer = p.add_tile(WMTSTileSource(url=url))
tile_renderer.alpha=1.0 if black_background else 0.15

InteractiveImage(p, image_callback, throttle=2000)

def image_callback2(x_range, y_range, w, h):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'meterswest', 'metersnorth')
    img = tf.interpolate(agg, cmap = reversed(Greys9))
    return tf.dynspread(img,threshold=0.75, max_px=8)

p = base_plot()
#InteractiveImage(p, image_callback2, throttle=1000)


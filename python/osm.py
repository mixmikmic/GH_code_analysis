import dask.dataframe as dd
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, visualize
import datashader as ds

df = dd.from_castra('data/osm.castra')
df.tail()

bound = 20026376.39
cvs = ds.Canvas(plot_width=1000, plot_height=1000,
                x_range=(-bound, bound), y_range=(-bound, bound))

with ProgressBar(), Profiler() as prof, ResourceProfiler(0.5) as rprof:
    agg = cvs.points(df, 'x', 'y', ds.count())

import datashader.transfer_functions as tf

tf.interpolate(agg, cmap=["lightcyan", "darkblue"], how="log")

tf.interpolate(agg.where(agg > 20), cmap=["lightcyan", "darkblue"], how="log")

from bokeh.io import output_notebook
from bokeh.resources import CDN
output_notebook(CDN, hide_banner=True)

visualize([prof, rprof])

from bokeh.plotting import figure, output_notebook
from bokeh.io import push_notebook
from datashader.bokeh_ext import InteractiveImage
from datashader import transfer_functions as tf

def create_image(x_range, y_range, w, h):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'x', 'y', ds.count())
    return tf.interpolate(agg.where(agg > 20), cmap=["lightcyan", "darkblue"], how="log")

p = figure(tools='pan,wheel_zoom,box_zoom,reset', plot_width=800, plot_height=800, 
           x_range=(-bound, bound), y_range=(-bound, bound))
           
p.axis.visible = False
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
#InteractiveImage(p, create_image, throttle=5000)


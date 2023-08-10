import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno

# read in the earthquake data from file, note that I only read
# in 5 columns with Origin time, Latitude, Longitude, Depth, and
# Magnitude of the earthquakes
df_eq = pd.read_csv('./data/catalog.csv', usecols=[0,1,2,3,4])

# now let's plot them using a black background
background = "black"
export = partial(export_image, background = background)
cm = partial(colormap_select, reverse=(background!="black"))

cvs = ds.Canvas(plot_width=1600, plot_height=1000)
agg = cvs.points(df_eq, 'Longitude', 'Latitude')
export(tf.interpolate(agg, cmap = cm(Hot,0.2), how='eq_hist'),"global_seismicity")


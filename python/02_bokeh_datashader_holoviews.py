import numpy as np
import pandas as pd
import xarray as xr

url_uncabled = 'https://opendap.oceanobservatories.org/thredds/dodsC/ooi/michaesm-marine-rutgers/20170725T195526-CE06ISSM-RID16-03-CTDBPC000-telemetered-ctdbp_cdef_dcl_instrument/deployment0006_CE06ISSM-RID16-03-CTDBPC000-telemetered-ctdbp_cdef_dcl_instrument.ncml'

ds = xr.open_dataset(url_uncabled, decode_times=True)
ds.data_vars

# Lets select these two variables to plot
x = 'time'
y = 'pressure'

df1 = ds[y].to_dataframe()
ds.close()

# Import Bokeh functions
import os
from bokeh.plotting import figure, output_file, reset_output, show, ColumnDataSource, save
from bokeh.models import BoxAnnotation
from bokeh.io import output_notebook # required to display Bokeh visualization in notebook

source = ColumnDataSource(
    data=dict(
        x=df1[x],
        y=df1[y],
    )
)

p = figure(width=600,
           height=400,
           title='CE06ISSM-RID16-03-CTDBPC000',
           x_axis_label='Time (GMT)',
           y_axis_label='Pressure (m)',
           x_axis_type='datetime')

p.line('x', 'y', line_width=3, source=source)
p.circle('x', 'y', fill_color='white', size=4, source=source)
output_notebook()
show(p)

url_cabled = 'https://opendap.oceanobservatories.org/thredds/dodsC/ooi/friedrich-knuth-rutgers/20180219T191719-RS03ASHS-MJ03B-10-CTDPFB304-streamed-ctdpf_optode_sample/deployment0003_RS03ASHS-MJ03B-10-CTDPFB304-streamed-ctdpf_optode_sample_20180205T190209.102547-20180219T190208.809978.nc'
ds = xr.open_dataset(url_cabled, decode_times=False)

cdf = ds[y].to_dataframe()

import datashader as dsp
import datashader.transfer_functions as tf

cvs = dsp.Canvas(plot_width=600, plot_height=400)
agg = cvs.line(cdf, x, y)

img = tf.shade(agg)
img

sampling = 1000
tf.shade(cvs.line(cdf[::sampling], 'time', 'pressure'))

import warnings
warnings.filterwarnings("ignore")

import holoviews as hv
from holoviews.operation.datashader import aggregate, datashade, dynspread, shade
from holoviews.operation import decimate
hv.notebook_extension('bokeh')

ds = xr.decode_cf(ds)
cdf2 = ds[y].to_dataframe()

from bokeh.models import DatetimeTickFormatter
def apply_formatter(plot, element):
    plot.handles['xaxis'].formatter = DatetimeTickFormatter()

get_ipython().run_cell_magic('opts', 'RGB [width=600]', '\ncurve = hv.Curve((cdf2[x], cdf2[y]))\ncurve')

get_ipython().run_cell_magic('opts', 'RGB [finalize_hooks=[apply_formatter] width=800]', '\n\ndatashade(curve, cmap=["blue"])')

get_ipython().run_cell_magic('opts', 'Overlay [finalize_hooks=[apply_formatter] width=800] ', '%%opts Scatter [tools=[\'hover\', \'box_select\']] (line_color="black" fill_color="red" size=10)\nfrom holoviews.operation.timeseries import rolling, rolling_outlier_std\nsmoothed = rolling(curve, rolling_window=50)\noutliers = rolling_outlier_std(curve, rolling_window=50, sigma=2)\n\ndatashade(curve, cmap=["blue"]) * dynspread(datashade(smoothed, cmap=["red"]),max_px=1) * outliers')




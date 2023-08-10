import numpy as np
import xarray as xr
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf
from cartopy import crs as ccrs

hv.extension('matplotlib', 'bokeh')
get_ipython().run_line_magic('output', 'size=200')

xr_ensembles = xr.open_dataset('/home/pangeo/data/ensembles.nc')
xr_ensembles

dataset = gv.Dataset(xr_ensembles, vdims='surface_temperature', crs=ccrs.PlateCarree())
dataset



hv.Dimension.type_formatters[np.datetime64] = '%Y-%m-%d'

geo_dims = ['longitude', 'latitude']
(dataset.to(gv.Image, geo_dims) * gf.coastline)[::5, ::5]

get_ipython().run_cell_magic('opts', "Points [color_index=2 size_index=None] (cmap='jet')", 'hv.Layout([dataset.to(el, geo_dims)[::10, ::5] * gf.coastline * gf.borders\n           for el in [gv.FilledContours, gv.LineContours, gv.Points]]).cols(1)')

dataset.to(gv.Image, geo_dims, dynamic=True) * gf.coastline

get_ipython().run_line_magic('output', "backend='bokeh'")

get_ipython().run_cell_magic('opts', "Curve [xrotation=25 width=600 height=400] {+framewise} NdOverlay [legend_position='right' toolbar='above']", "dataset.to(hv.Curve, 'time', dynamic=True).overlay('realization')")

get_ipython().run_cell_magic('opts', "HeatMap [width=600 colorbar=True tools=['hover'] toolbar='above']", "dataset.to(hv.HeatMap, ['realization', 'time'], dynamic=True)")

get_ipython().run_cell_magic('output', "backend='matplotlib'", "hv.Layout([dataset.to(hv.Violin, d, groupby=[], datatype=['dataframe']).options(xrotation=25)\n           for d in ['time', 'realization']])")

get_ipython().run_cell_magic('output', "backend='matplotlib'", 'northern = dataset.select(latitude=(25, 75))\n(northern.select(longitude=(260, 305)).to(gv.Image, geo_dims) *\n northern.select(longitude=(330, 362)).to(gv.Image, geo_dims) *\n gf.coastline)[::5, ::5]')

get_ipython().run_cell_magic('opts', "NdOverlay [width=600 height=400 legend_position='right' toolbar='above'] Curve (color=Palette('Set1'))", "dataset.select(latitude=0, longitude=0).to(hv.Curve, ['time']).reindex().overlay()")

hv.Spread(dataset.aggregate('latitude', np.mean, np.std)) +hv.Spread(dataset.aggregate('longitude', np.mean, np.std))


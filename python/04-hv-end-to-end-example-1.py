from holoext.bokeh import Mod
import numpy as np
import pandas as pd
import holoviews as hv
import warnings

warnings.filterwarnings('ignore')  # bokeh deprecation warnings
hv.extension('bokeh')

df = pd.read_parquet('../datasets/weather_station_data.parquet')
df.head()

ds = hv.Dataset(df, kdims=['station', 'date', 'year', 'month'])

ds

curve = ds.to(hv.Curve, kdims=['date'], vdims=['precip_cumsum_in', 'precip_in'], groupby='station')
Mod(xlabel='Date', ylabel='Cumulative Precip [in]').apply(curve)

STATIONS = df.station.unique() # a list of unique stations
STATIONS

curve_list = []
for station in STATIONS:
    curve = hv.Curve(
        ds.select(station=station),
        kdims=['date'],
        vdims=['precip_cumsum_in', 'precip_in'],
        label=station)
    curve = curve.opts(style=dict(line_alpha=0.75))  # style the curves
    curve_list.append(curve)

curves_overlay = hv.Overlay(curve_list)
Mod(xlabel='Date',
    ylabel='Cumulative Precip [in]',
    tools=['ypan', 'hover', 'ywheel_zoom', 'save', 'reset']).apply(curves_overlay)

max_hmap = ds.to(
    hv.HeatMap,
    kdims=['year', 'month'],
    vdims=['max_temp_f'],
    groupby='station',
    label='Max Temp [F]').redim.range(max_temp_f=(10, 110))

min_hmap = ds.to(
    hv.HeatMap,
    kdims=['year', 'month'],
    vdims=['min_temp_f'],
    groupby='station',
    label='Min Temp [F]').redim.range(min_temp_f=(0, 80))

hmaps = max_hmap + min_hmap

Mod(xlabel='Year', ylabel='Month', width=1500).apply(hmaps).cols(1)

# define plot and style options for different elements
scatter_opts = dict(width=500, height=500)
scatter_style = dict(alpha=0.35, size=5, color='red')
hist_style = dict(alpha=0.75, line_color=None)

opts = {
    'Scatter': {
        'plot': scatter_opts,
        'style': scatter_style
    },
    'Histogram': {
        'style': hist_style
    }
}

scatter = ds.to(
    hv.Scatter,
    kdims=['avg_wind_speed_kts'],
    vdims=['avg_wind_drct'],
    groupby='station')

scatter_hists = scatter.hist(
    num_bins=100, dimension=['avg_wind_speed_kts',
                           'avg_wind_drct']).opts(opts).redim.label(
                               avg_wind_speed_kts='Wind Speed [kts]',
                               avg_wind_drct='Wind Dir. [deg]',
                               avg_wind_speed_kts_frequency='Freq.',
                               avg_wind_drct_frequency='Freq.')


Mod().apply(scatter_hists)

points = ds.to(
    hv.Points,
    kdims=['date', 'max_temp_f'],
    vdims=['precip_in'],
    groupby='station',
    group='Temperature and Precipitation for').opts(
        style={'Points': dict(alpha=0.35)})

Mod(tools=['hover', 'save', 'ypan', 'ywheel_zoom'],
    ylabel='Max Temp [F]',
    xlabel='Date',
    width=1000,
    num_xticks=15,
    xrotation=35,
    size_index=2,
    color_index=2,
    scaling_factor=25,
    colorbar_n=5,
    colorbar_title='[in]').apply(points).redim.range(precip_in=(0, 5))

def make_box(month, year):
    """Creates and returns a box and whisker plot for given month and year"""
    sub_ds = ds.select(month=month, year=year)
    text = hv.Text('', 45, '{0:2d}/{1:s}'.format(
        month,
        str(year)[2:])).opts(
            style=dict(text_alpha=0.5, text_font_size='12px'))
    box = hv.BoxWhisker(sub_ds, 'station', 'max_rh')
    return box * text


boxes = {
    (month, year): make_box(month, year)
    for month in df['month'].unique()[
        2::4]  # to speed it up the computation, select a few months
    for year in df['year'].unique()[-4:]  # Slice the last 4 years
}

gridmatrix = hv.GridMatrix(boxes)
title = 'Max Relative Humidity Spread'
Mod(
    axiswise=False,
    xlabel='Station',
    ylabel='Max RH [%]',
    autosize=True,
    title_format=title,
    label_scaler=0.6,
    merge_tools=True,
    tools=['save', 'hover'],
    logo=False,
    plot_size=250  # width/height doesn't work in grid matrix
).apply(gridmatrix)


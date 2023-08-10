# Import packages

import pandas as pd
import folium

import math
import json

from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LogColorMapper,
    LinearColorMapper,
    NumeralTickFormatter,
    LogTicker,
    ColorBar
)
from bokeh.palettes import RdYlBu11 as palette1
from bokeh.palettes import OrRd7 as palette2
from bokeh.plotting import figure, output_notebook, show
output_notebook()

permits = pd.read_csv('permits.csv')
permits = permits[permits['variable'] == 'Single Family']

permits_by_state = permits.groupby(['state','year'])['value'].sum().reset_index()

state_permits_summed = permits_by_state.groupby('year').sum().reset_index()

state_permits_pivoted = (permits_by_state[permits_by_state['year'].isin([2005,2009])]
     .pivot(index='state', columns='year', values='value')
     .reset_index())
state_permits_pivoted['delta'] = state_permits_pivoted['delta'] = state_permits_pivoted[2005] - state_permits_pivoted[2009]

# Sum up permit data for each county in MO
mo_permits = permits[permits['StateAbbr'] == 'MO']

mo_permits_by_county = mo_permits.groupby(['county','year']).sum().reset_index()

mo_permits_summed = mo_permits.groupby('year').sum().reset_index()

mo_permits_pivoted = (mo_permits_by_county[mo_permits_by_county['year'].isin([2005, 2009])]
                      .pivot(index='county', columns='year', values='value')
                      .reset_index())
mo_permits_pivoted['delta'] = mo_permits_pivoted[2005] - mo_permits_pivoted[2009]
mo_permits_pivoted.dropna(inplace=True)

# US Timeseries
us_timeseries = figure(plot_width=800, plot_height=500, title='US Single Family Building Permits')
us_timeseries.xaxis.axis_label = 'Year'
us_timeseries.yaxis.axis_label = 'Number of Permits'
us_timeseries.title.text_font_size = '20pt'

us_timeseries.line(state_permits_summed['year'], state_permits_summed['value'],
                   line_width=4, line_color='red')

show(us_timeseries)

state_permits_pivoted.columns = ['state', 2005, 2009, 'delta']

map1 = folium.Map([43,-100], zoom_start=4)

map1.choropleth(
    open('USA_adm1 (2).json').read(),
    data=state_permits_pivoted,
    columns=['state', 'delta'],
    key_on='properties.ID_1',
    fill_color='OrRd',
    )
map1

# Missouri Timeseries
mo_timeseries = figure(plot_width=800, plot_height=500, title='Missouri Single Family Building Permits')
mo_timeseries.xaxis.axis_label = 'Year'
mo_timeseries.yaxis.axis_label = 'Number of Permits'
mo_timeseries.title.text_font_size = '20pt'

mo_timeseries.line(mo_permits_summed['year'], mo_permits_summed['value'],
                   line_width=4, line_color='red')

show(mo_timeseries)

mo_json = json.load(open('tl_2010_29_county10.json'))
mo_permits_pivoted.columns = ['COUNTYFP10', 2005, 2009, 'delta']
mo_json = json.load(open('tl_2010_29_county10.json'))
for county in mo_json['features']:
    county['properties']['COUNTYFP10'] = int(county['properties']['COUNTYFP10'])

# Missouri Heatmap

map2 = folium.Map(location=[38.5,-93], 
                  zoom_start=6.5,
                  tiles='Mapbox Bright')

map2.choropleth(
    open('tl_2010_29_county10.json').read(),
    data=mo_permits_pivoted,
    columns=['COUNTYFP10', 'delta'],
    key_on='properties.COUNTYFP10',
    fill_color='OrRd',
    highlight=True)

map2


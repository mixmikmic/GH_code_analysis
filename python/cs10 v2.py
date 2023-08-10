# Import packages

import pandas as pd
#import geopandas as gpd

import math

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

# Import data

from bokeh.sampledata.us_counties import data as counties
from bokeh.sampledata.us_states import data as states

permits = pd.read_csv('permits.csv')
permits = permits[permits['variable'] == 'Single Family']

# Transform data to include geographic data

# Transform state and city data to a more useful dict for plotting
states = {
    code: state for code, state in states.items() if (not state['name'] == 'Alaska' and 
                                                      not state['name'] == 'Hawaii' and 
                                                      not state['name'] == 'Puerto Rico')
}
state_xs = [states[state]['lons'] for state in sorted(states.keys())]
state_ys = [states[state]['lats'] for state in sorted(states.keys())]

# Sum up permit data for each state
state_permits = pd.DataFrame(permits.groupby(['StateAbbr', 'year', 'variable'])['value']
                                    .sum()).reset_index().rename(columns = {'StateAbbr':'state'})
state_permits_filtered = state_permits[~(state_permits['state'].isin(['AK', 'HI', 'PR']))].copy().reset_index(drop=True)

state_permits_summed = pd.DataFrame(state_permits_filtered.groupby('year')['value'].sum()).reset_index()

state_permits_pivoted = (state_permits_filtered[state_permits_filtered['year'].isin([2006, 2009])]
                         .pivot(index='state', columns='year', values='value')
                         .reset_index())
state_permits_pivoted['delta'] = state_permits_pivoted[2006] - state_permits_pivoted[2009]

# Sum up permit data for each county in MO
mo_permits = permits[permits['StateAbbr'] == 'MO'].copy()
mo_permits_summed = pd.DataFrame(mo_permits.groupby('year')['value'].sum()).reset_index()

mo_permits_pivoted = (mo_permits[mo_permits['year'].isin([2006, 2009])]
                      .pivot(index='countyname', columns='year', values='value')
                      .reset_index())
mo_permits_pivoted['delta'] = mo_permits_pivoted[2006] - mo_permits_pivoted[2009]

mo_counties = {
    code: county for code, county in counties.items() if (county["state"] == "mo" and
                                                          county['detailed name'].split(',')[0] in list(mo_permits_pivoted['countyname']))
}
county_xs = [mo_counties[county]['lons'] for county in sorted(mo_counties.keys())]
county_ys = [mo_counties[county]['lats'] for county in sorted(mo_counties.keys())]

# US Timeseries
us_timeseries = figure(plot_width=800, plot_height=500, title='US Single Family Building Permits')
us_timeseries.xaxis.axis_label = 'Year'
us_timeseries.yaxis.axis_label = 'Number of Permits'
us_timeseries.title.text_font_size = '20pt'

us_timeseries.line(state_permits_summed['year'], state_permits_summed['value'],
                   line_width=4, line_color='red')

show(us_timeseries)

# US Heatmap
delta = [delta for delta in list(state_permits_pivoted['delta'])]
name = [name for name in list(state_permits_pivoted['state'])]
source = ColumnDataSource(dict(
    x=state_xs,
    y=state_ys,
    name=name,
    delta=delta
))
color_mapper = LogColorMapper(palette=palette1, low=334, high=167912)

us_heatmap = figure(title='Decrease in Building Permits 2006-2009',
                    plot_width=950, plot_height=600,
                    x_axis_location=None, y_axis_location=None,
                    tools='pan, wheel_zoom, save, reset')
us_heatmap.title.text_font_size = '20pt'
us_heatmap.patches('x', 'y', source=source,
                   fill_color={'field': 'delta', 'transform': color_mapper},
                   line_color="#e8e8e8", line_width=1)

hover = HoverTool(tooltips=[
    ("State", "@name"),
    ("Decrease", "@delta permits")
])

us_heatmap.add_tools(hover)

color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                     formatter=NumeralTickFormatter(),
                     label_standoff=20, border_line_color=None, location=(0,0))

us_heatmap.add_layout(color_bar, 'right')
us_heatmap.toolbar_location = 'above'

show(us_heatmap)

# Missouri Timeseries
mo_timeseries = figure(plot_width=800, plot_height=500, title='Missouri Single Family Building Permits')
mo_timeseries.xaxis.axis_label = 'Year'
mo_timeseries.yaxis.axis_label = 'Number of Permits'
mo_timeseries.title.text_font_size = '20pt'

mo_timeseries.line(mo_permits_summed['year'], mo_permits_summed['value'],
                   line_width=4, line_color='red')

show(mo_timeseries)

# Missouri Heatmap
counties_df = pd.DataFrame(mo_counties).transpose()
counties_df['detailed name'] = counties_df['detailed name'].apply(lambda x: x.split(',')[0])
merged = pd.merge(counties_df, mo_permits_pivoted, left_on='detailed name', right_on='countyname').fillna(0)

mo_source = ColumnDataSource(dict(
    x=list(merged['lons']),
    y=list(merged['lats']),
    name=list(merged['detailed name']),
    delta=list(merged['delta'])
))

color_mapper = LinearColorMapper(palette=palette2[4::-1], low=0, high=4784)

mo_heatmap = figure(title='Decrease in Missouri Building Permits 2006-2009',
                    plot_width=950, plot_height=600,
                    x_axis_location=None, y_axis_location=None,
                    tools='pan, wheel_zoom, save, reset')
mo_heatmap.title.text_font_size = '20pt'
mo_heatmap.grid.grid_line_alpha = 0
mo_heatmap.patches('x', 'y', source=mo_source,
                  fill_color={'field': 'delta', 'transform': color_mapper},
                  line_color="#e8e8e8", line_width=1)

hover = HoverTool(tooltips=[
   ("State", "@name"),
   ("Decrease", "@delta permits")
])

mo_heatmap.add_tools(hover)

color_bar = ColorBar(color_mapper=color_mapper)

us_heatmap.add_layout(color_bar, 'right')
mo_heatmap.toolbar_location = 'above'

show(mo_heatmap)


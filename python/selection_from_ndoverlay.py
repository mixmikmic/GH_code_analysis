import pandas as pd
import numpy as np 
import numpy as np
import holoviews as hv
from holoviews import streams
hv.extension('bokeh')

macro_df = pd.read_csv('http://assets.holoviews.org/macro.csv', '\t')

key_dimensions   = [('year', 'Year'), ('country', 'Country')]
value_dimensions = [('unem', 'Unemployment'), ('capmob', 'Capital Mobility'),
                    ('gdp', 'GDP Growth'), ('trade', 'Trade')]
macro = hv.Table(macro_df, key_dimensions, value_dimensions)


get_ipython().run_line_magic('opts', "Scatter [width=700 height=400 scaling_method='width' scaling_factor=2 size_index=2 show_grid=True tools=['box_select', 'lasso_select', 'tap']]")
get_ipython().run_line_magic('opts', "Scatter (color=Cycle('Category20') line_color='k')")
get_ipython().run_line_magic('opts', "NdOverlay [legend_position='left' show_frame=False]")
gdp_unem_scatter = macro.to.scatter('Year', ['GDP Growth', 'Unemployment'])
overlayed_scatter = gdp_unem_scatter.overlay('Country')

sel  = streams.Selection1D(source=overlayed_scatter)

def selection_callback(index):
    divtext = ""
    for i in index:
        divtext += f'{i} {macro_df["country"][i]}<p>'
    return hv.Div(divtext)

div = hv.DynamicMap(selection_callback, streams=[sel])

overlayed_scatter 

div

macro_df = pd.read_csv('http://assets.holoviews.org/macro.csv', '\t')

# Read data frame with giffile links and stick on another column to the economic data
df = pd.read_csv("df.csv")
macro_df["giffile"]=df.giffile

key_dimensions   = [('year', 'Year'), ('country', 'Country')]
value_dimensions = [('unem', 'Unemployment'), ('capmob', 'Capital Mobility'),
                    ('gdp', 'GDP Growth'), ('trade', 'Trade'), ('giffile','giffile')]

options = dict(
    color_index='Country', legend_position='left', width=700, height=400,
    scaling_method='width', scaling_factor=2, size_index=2, show_grid=True,
    tools=['box_select'], line_color='k', cmap='Category20'
)

macro = hv.Table(macro_df, key_dimensions, value_dimensions)
gdp_unem_scatter = macro.to.scatter('Year', ['GDP Growth', 'Unemployment', 'Country']).options(**options)
sel  = streams.Selection1D(source=gdp_unem_scatter)

def selection_callback(index):
    divtext=""
    for i in index:
        # Create HTML
        divtext += f'{macro_df["year"][i]}  <img src="{macro_df["giffile"][i]}" width=150> <p>'
    return hv.Div(str(divtext))


div = hv.DynamicMap(selection_callback, streams=[sel])
# this line creates an empty selection event in the the selection stream when the plot is reset
hv.streams.PlotReset(source=gdp_unem_scatter, subscribers=[lambda reset: sel.event(index=[])])
gdp_unem_scatter << div

hv.__version__




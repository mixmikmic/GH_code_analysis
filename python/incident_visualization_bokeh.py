import os
import numpy as np
import pandas as pd

from bokeh.io import show
from bokeh.io import output_notebook


from bokeh.models import Plot
from bokeh.models import Range1d
from bokeh.models import LinearAxis
from bokeh.models import SingleIntervalTicker
from bokeh.models import ColumnDataSource

from bokeh.palettes import Spectral6
from bokeh.charts import BoxPlot

from bokeh.layouts import layout
from bokeh.layouts import widgetbox

from bokeh.embed import file_html


from bokeh.models import Text
from bokeh.models import Slider
from bokeh.models import Circle
from bokeh.models import CustomJS
from bokeh.models import HoverTool
from bokeh.models import LinearAxis

output_notebook()

regiondf = pd.read_csv(os.path.abspath('C:\project\ship-happens\data//bokeh_region.txt'), sep='\t')
dead = pd.read_csv(os.path.abspath('C:\project\ship-happens\data//bokeh_dead.txt'), sep='\t')
injured = pd.read_csv(os.path.abspath('C:\project\ship-happens\data//bokeh_injured.txt'), sep='\t')
accident = pd.read_csv(os.path.abspath('C:\project\ship-happens\data//bokeh_accident.txt'), sep='\t')
accident.head()

dead.fillna(value=0, inplace=True)
injured.fillna(value=0, inplace=True)
accident.fillna(value=0, inplace=True)
regiondf.set_index('Country', inplace=True )
injured.set_index('Country', inplace=True )
dead.set_index('Country', inplace=True )
accident.set_index('Country', inplace=True )

regiondf.head()

#accident.describe()

def process_data():    
    # Make the column names ints not strings for handling    
    columns = list(injured.columns)
    years = list(range(int(columns[0]), int(columns[-1])))
    rename_dict = dict(zip(columns, years))
    
    binjured = injured.rename(columns=rename_dict)
    bdead = dead.rename(columns=rename_dict)
    regions = regiondf.rename(columns=rename_dict)
    baccident = accident.rename(columns=rename_dict)

    # Turn total vessel into bubble sizes. Use min_size and factor to tweak.
    scale_factor = .2
    baccident_size = np.sqrt((baccident) / np.pi) / scale_factor
    min_size = .1
    baccident_size = baccident_size.where(baccident_size >= min_size).fillna(min_size)
    
    # Use pandas categories and categorize & color the regions
    regions.Group = regions.Group.astype('category')
    regions_list = list(regions.Group.cat.categories)
   
    def get_color(r):
        return Spectral6[regions_list.index(r.Group)]
    regions['region_color'] = regions.apply(get_color, axis=1)   
   
    
    return binjured, bdead, baccident_size, regions, years, regions_list 

injured_df, dead_df, baccident_df, regions_df, years, regions = process_data()

sources = {}

region_color = regions_df['region_color']
region_color.name = 'region_color'

for year in years:
    sinjured = injured_df[year]
    sinjured.name = 'totalinjured'
    sdead = dead_df[year]
    sdead.name = 'totaldead' 
    taccident = baccident_df[year]
    taccident.name = 'totalaccident' 
    new_df = pd.concat([sinjured, sdead, taccident, region_color], axis=1)
    sources['_' + str(year)] = ColumnDataSource(new_df)  
#print(regions)

#print(new_df)

dictionary_of_sources = dict(zip([x for x in years], ['_%s' % x for x in years]))

js_source_array = str(dictionary_of_sources).replace("'", "")
#js_source_array

xdr = Range1d(-5, 750)
ydr = Range1d(-3, 150)
plot = Plot(
    x_range=xdr,
    y_range=ydr,
    plot_width=800,
    plot_height=500,
    outline_line_color=None,
    toolbar_location=None, 
    min_border=20,
)

show(plot)

AXIS_FORMATS = dict(
    minor_tick_in=None,
    minor_tick_out=None,
    major_tick_in=None,
    major_label_text_font_size="10pt",
    major_label_text_font_style="normal",
    axis_label_text_font_size="14pt",

    axis_line_color='#AAAAAA',
    major_tick_line_color='#AAAAAA',
    major_label_text_color='#666666',

    major_tick_line_cap="round",
    axis_line_cap="round",
    axis_line_width=1,
    major_tick_line_width=1,
)

xaxis = LinearAxis(ticker=SingleIntervalTicker(interval=30), axis_label="Total Injury", **AXIS_FORMATS)
yaxis = LinearAxis(ticker=SingleIntervalTicker(interval=10), axis_label="Total Death", **AXIS_FORMATS)   
plot.add_layout(xaxis, 'below')
plot.add_layout(yaxis, 'left')

show(plot)

text_source = ColumnDataSource({'year': ['%s' % years[0]]})
text = Text(x=100, y=15, text='year', text_font_size='150pt', text_color='#EEEEEE')
plot.add_glyph(text_source, text)

show(plot)

# Add the circle
renderer_source = sources['_%s' % years[0]]
circle_glyph = Circle(
    x='totalinjured', y='totaldead', size='totalaccident',
    fill_color='region_color', fill_alpha=0.8, 
    line_color='#7c7e71', line_width=0.5, line_alpha=0.5)

circle_renderer = plot.add_glyph(renderer_source, circle_glyph)

# Add the hover (only against the circle and not other plot elements)
tooltips = "@index"
plot.add_tools(HoverTool(tooltips=tooltips, renderers=[circle_renderer]))

show(plot)

text_x = 30
text_y = 140
for i, region in enumerate(regions):
    plot.add_glyph(Text(x=text_x, y=text_y, text=[region], text_font_size='10pt', text_color='#666666'))
    plot.add_glyph(Circle(x=text_x - 9, y=text_y+ 3, fill_color=Spectral6[i], size=10, line_color=None, fill_alpha=0.8))
    text_y = text_y - 6

show(plot)

# Add the slider
code = """
    var year = slider.get('value'),
        sources = %s,
        new_source_data = sources[year].get('data');
    renderer_source.set('data', new_source_data);
    text_source.set('data', {'year': [String(year)]});
""" % js_source_array

callback = CustomJS(args=sources, code=code)
slider = Slider(start=years[0], end=years[-1], value=1, step=1, title="Year", callback=callback)
callback.args["renderer_source"] = renderer_source
callback.args["slider"] = slider
callback.args["text_source"] = text_source

show(widgetbox(slider))

show(layout([[plot], [slider]], sizing_mode='scale_width'))




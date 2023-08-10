
# This whole thing takes about 300 seconds
from bokeh.models import Plot
from bokeh.models import LogScale

from bokeh.models import LinearAxis
from bokeh.models import SingleIntervalTicker
from bokeh.models import LogAxis
from bokeh.models import LogTicker
from bokeh.models import Range1d

from bokeh.models import Text
from bokeh.models import ColumnDataSource

from bokeh.models import Circle
from bokeh.palettes import Spectral6

from bokeh.io import export_png

import os
import imageio
import pandas as pd
import time
import math
import numpy as np

encoding = ['latin1', 'iso8859-1', 'utf-8'][1]


# Set up fertility data
fertility_df = pd.read_csv('./data/csv/fertility_df.csv', encoding=encoding)
fertility_df.set_index(keys='Country', inplace=True)
c = fertility_df.columns
missing_years = [i for i in [str(i) for i in range(int(c[0]), int(c[-1]))] if i not in c]
if len(missing_years):
    for year in missing_years:
        fertility_df.loc[:, year] = pd.Series(None, index=fertility_df.index)
    fertility_df = fertility_df.T.sort_index(inplace=False).interpolate().T
    fertility_df.to_csv('./data/csv/fertility_df.csv', index=True)


# Set up income data
income_df = pd.read_csv('./data/csv/income_df.csv', encoding=encoding)
income_df.set_index(keys='Country', inplace=True)
c = income_df.columns
missing_years = [i for i in [str(i) for i in range(int(c[0]), int(c[-1]))] if i not in c]
if len(missing_years):
    for year in missing_years:
        income_df.loc[:, year] = pd.Series(None, index=income_df.index)
    income_df = income_df.T.sort_index(inplace=False).interpolate().T
    income_df.to_csv('./data/csv/income_df.csv', index=True)


# Set up population data
population_df = pd.read_csv('./data/csv/population_df.csv', encoding=encoding)
population_df.set_index(keys='Country', inplace=True)
c = population_df.columns
missing_years = [i for i in [str(i) for i in range(int(c[0]), int(c[-1]))] if i not in c]
if len(missing_years):
    for year in missing_years:
        population_df.loc[:, year] = pd.Series(None, index=population_df.index)
    population_df = population_df.T.sort_index(inplace=False).interpolate().T
    population_df.to_csv('./data/csv/population_df.csv', index=True)

# Turn population into bubble sizes. Use min_size and factor to tweak
scale_factor = 400
population_size_df = np.sqrt(population_df / np.pi) / scale_factor
min_size = 3
population_size_df = population_size_df.where(population_size_df >= min_size).fillna(min_size)


# Set up regions data
regions_df = pd.read_csv('./data/csv/regions_df.csv', encoding=encoding)
regions_df['Group'] = regions_df['Group'].astype('category')
regions_list = list(regions_df['Group'].cat.categories)
regions_df.set_index(keys='Country', inplace=True)

def get_color(r):
    return Spectral6[regions_list.index(r['Group'])]

regions_df['region_color'] = regions_df.apply(get_color, axis=1)

region_color_series = regions_df['region_color']
region_color_series.name = 'region_color'

columns_list = list(fertility_df.columns)


# Set up plot formats
PLOT_FORMATS = dict(
    plot_width=800,
    plot_height=400,
    outline_line_color=None,
    toolbar_location=None, 
    min_border=20,
)

# Set up axis of plots
AXIS_FORMATS = dict(
    minor_tick_in=None,
    minor_tick_out=None,
    major_tick_in=None,
    major_label_text_font_size="10pt",
    major_label_text_font_style="normal",
    axis_label_text_font_size="10pt",

    axis_line_color='#AAAAAA',
    major_tick_line_color='#AAAAAA',
    major_label_text_color='#666666',

    major_tick_line_cap="round",
    axis_line_cap="round",
    axis_line_width=1,
    major_tick_line_width=1,
)

# Children per woman (total fertility)
x_low = int(math.floor(fertility_df.min().min()))
x_high = int(math.ceil(fertility_df.max().max()))
x_data_range = Range1d(x_low-0.5*x_low, x_high+0.1*x_high)
x_interval = int(math.ceil((x_high - x_low) / 9))

# Personal income (GDP per capita)
y_low = int(math.floor(income_df.min().min()))
y_high = int(math.ceil(income_df.max().max()))
y_data_range = Range1d(y_low-0.5*y_low, y_high+0.5*y_high)
y_interval = int(math.ceil((y_high - y_low) / 5))


# Set up background text formats
BACKGROUND_FORMATS = dict(
    x=1,
    y=150,
    text='year',
    text_font_size='150pt',
    text_color='#EEEEEE',
)

# Set up circle formats
CIRCLE_FORMATS = dict(
    x='fertility',
    y='income',
    size='population',
    fill_color='region_color',
    fill_alpha=0.8,
    line_color='#7c7e71',
    line_width=0.5,
    line_alpha=0.5,
)

# Set up legend text formats
LEGEND_TEXT_FORMATS = dict(
    text_font_size='10pt',
    text_color='#666666',
)

# Set up legend CIRCLE formats
LEGEND_CIRCLE_FORMATS = dict(
    size=10,
    line_color=None,
    fill_alpha=0.8,
)


# Create and save images to the png folder
t0 = time.time()
png_dir = "./saves/png/"
for year in columns_list:
    if (year in fertility_df) and (year in income_df) and (year in population_size_df):
        
        # Concatonate the 3 year-series with the region color series
        fertility_series = fertility_df[year]
        fertility_series.name = 'fertility'
        income_series = income_df[year]
        income_series.name = 'income'
        population_series = population_size_df[year]
        population_series.name = 'population'
        new_df = pd.concat([fertility_series, income_series, population_series, region_color_series], axis=1)

        # Build the plot
        plot = Plot(x_range=x_data_range, y_range=y_data_range, y_scale=LogScale(), **PLOT_FORMATS)

        # Add the axes
        xaxis = LinearAxis(ticker=SingleIntervalTicker(interval=x_interval), 
                           axis_label="Children per woman (total fertility)", 
                           **AXIS_FORMATS)
        plot.add_layout(xaxis, 'below')
        #yaxis = LogAxis(ticker=LogTicker(), axis_label="Personal income (GDP per capita)", **AXIS_FORMATS)
        yaxis = LinearAxis(ticker=LogTicker(), axis_label="Personal income (GDP per capita)", **AXIS_FORMATS)
        plot.add_layout(yaxis, 'left')

        # Add the background year text
        text_source = ColumnDataSource({'year': ['%s' % year]})
        text = Text(**BACKGROUND_FORMATS)
        plot.add_glyph(text_source, text)

        # Add the circles
        renderer_source = ColumnDataSource(new_df)
        circle_glyph = Circle(**CIRCLE_FORMATS)
        circle_renderer = plot.add_glyph(renderer_source, circle_glyph)

        # Add the legend
        text_x = 7
        for i, region in enumerate(regions_list):
            plot.add_glyph(Text(x=text_x, y=int(10**(-0.15*i+5)), text=[region], **LEGEND_TEXT_FORMATS))
            plot.add_glyph(Circle(x=text_x - 0.1, y=int(10**(-0.15*i+5.08)), fill_color=Spectral6[i], **LEGEND_CIRCLE_FORMATS))

        # Save as PNG
        export_png(plot, filename=png_dir+"plot_%s.png" % year)

t1 = time.time()
print(t1-t0, time.ctime(t1))


# Concatonate images into movie
t0 = time.time()
images = []
for subdir, dirs, files in os.walk(png_dir):
    for file in files:
        filepath = os.path.join(subdir, file)
        if filepath.endswith(".png"):
            images.append(imageio.imread(filepath))
imageio.mimsave('./saves/gif/income_vs_fertility_movie.gif', images)
t1 = time.time()
print(t1-t0, time.ctime(t1))




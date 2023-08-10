import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.palettes import Spectral6

df = pd.read_csv('../../data/imf_indicators.tsv', delimiter='\t', thousands=',', na_values=['n/a'],
                 encoding='ISO-8859-1')

df.head()

df['Subject Descriptor'].value_counts()

def mscatter(chart, x, y, typestr):
    chart.scatter(x, y, marker=typestr, line_color="#6666ee",
                  fill_color="#ee6666", fill_alpha=0.5, size=12)

def draw_scatter(df):
    chart = figure(title="IMF Unemployment")
    output_notebook()
    for year in ['2013', '2014', '2015', '2016']:
        df[year].map(lambda x: mscatter(chart, int(year), x, 'circle'))
    return chart

chart = draw_scatter(df[df['Subject Descriptor'] == 'Unemployment rate'])
show(chart)

grouped_mean_df = df[df['Subject Descriptor'] == 'Unemployment rate'].groupby(
    'Country')[['2013', '2014', '2015', '2016']].mean()

grouped_mean_df.head()

def draw_line(df):
    chart = figure(title="IMF Unemployment")
    output_notebook()
    color_count = 0
    for country, data in df.iterrows():
        chart.line(data.index.values, data.values, legend=country, line_color=Spectral6[color_count])
        if color_count+1 != len(Spectral6):
            color_count += 1
        else:
            color_count = 0
    return chart

chart = draw_line(grouped_mean_df)

show(chart)

get_ipython().magic('load solutions/visualization_solution.py')




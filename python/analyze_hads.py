from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
if (code_show){
$('div.input').hide();
} else {
$('div.input').show();
}
code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')

import glob
import os

import numpy as np
import pandas as pd

import ipywidgets

import bokeh.plotting
import bokeh.layouts
import bokeh.models
import bokeh.io
from bokeh.palettes import Category10_10 as palette

bokeh.plotting.output_notebook()

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

get_ipython().system(' ls -lh data/')

fnames = glob.glob('data/*txt')
fnames.sort()

fnames



years = np.array([int(fname.strip('data/hads.txt')) for fname in fnames])

# print(years)  # too much data to start with 
print(years[::3])  # lets some years

data = {}
columns = []
for year in years[::3]:
    fname = 'data/hads{}.txt'.format(year)
    assert os.path.exists, 'check input, no such file path/name: {}'.format(fname)
    data[year] = pd.read_csv(fname, skiprows=0, sep=',')
    data[year].columns = map(str.lower, data[year].columns)  # make column labels lowercase
    data[year]['year'] = year  # add the year as a feature
    columns.append(data[year].columns)
    

years = years[::3]

columns[0][1] in columns[-1]

# how similar are the columns?
for i, i_columns in enumerate(columns):
    for column in i_columns:
        if column not in columns[-1]:
            print('`{}` values, from {} data, is not in the 2009 data'.format(column, years[i]))

df_raw = pd.concat([data[key] for key in data.keys()])

df_raw.head()

df = df_raw.dropna(axis=1, how='any')

df.head()

int(np.where(years == 1997)[0])

colors = df.year.apply(lambda year: palette[int(np.where(years == year)[0])])

df = df.assign(color=pd.Series(colors, index=df.index))

year = 1985

def update(x='bedrms', y='cost06'):
    r.data_source.data['x'] = df[df.year == year][x]
    r.data_source.data['y'] = df[df.year == year][y]
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y
    bokeh.io.push_notebook()

p = bokeh.plotting.figure(x_axis_label='bedrms', y_axis_label='cost06')
x = df[df.year == year].bedrms
y = df[df.year == year].cost06
colors = df[df.year == year].color
r = p.circle(x, y, color=colors, alpha=0.2)
bokeh.plotting.show(p, notebook_handle=True)

ipywidgets.interact(update, x=list(df.columns), y=list(df.columns))

# find the quartile and IQR for each category by year
df_by_year = df.groupby('year')

q1 = df_by_year.quantile(q=0.25)
q2 = df_by_year.quantile(q=0.5)
q3 = df_by_year.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
upper = q1 - 1.5 * iqr

# get the range for the stems
qmin = df_by_year.quantile(q=0.0)
qmax = df_by_year.quantile(q=1.0)

# make string version for x-labels
yrs_str = [str(year) for year in years]

def update_box(feature='cost06'):
    # update stems
    s1.data_source.data['y0'] = qmin[feature]
    s1.data_source.data['y1'] = q1[feature]
    s2.data_source.data['y0'] = q3[feature]
    s2.data_source.data['y1'] = qmax[feature]
    
    # update boxes
    v1.data_source.data['bottom'] = q1[feature]
    v1.data_source.data['top'] = q2[feature]
    v2.data_source.data['bottom'] = q2[feature]
    v2.data_source.data['top'] = q3[feature]

    # update whiskers
    r1.data_source.data['y'] = qmin[feature]
    r2.data_source.data['y'] = qmax[feature]

    p.yaxis.axis_label = feature
    bokeh.io.push_notebook()

initial = 'cost06'
p = bokeh.plotting.figure(x_range=yrs_str, y_axis_label=initial)

# stems
s1 = p.segment(yrs_str, qmin[initial], yrs_str, q1[initial], line_color='black')
s2 = p.segment(yrs_str, q3[initial], yrs_str, qmax[initial], line_color='black')

# boxes
b_width = 0.7
v1 = p.vbar(yrs_str, b_width, q2[initial], q1[initial], fill_color=palette[1], line_color='black')
v2 = p.vbar(yrs_str, b_width, q3[initial], q2[initial], fill_color=palette[0], line_color='black')

# whiskers
w_width = 0.2
r1 = p.rect(yrs_str, qmin[initial], w_width, 0.001, fill_color='black', line_color='black')
r2 = p.rect(yrs_str, qmax[initial], b_width, 0.001, fill_color='black', line_color='black')

p.xgrid.grid_line_color = None

p.xaxis.major_label_text_font_size = "12pt"

bokeh.plotting.show(p, notebook_handle=True)

ipywidgets.interact(update_box, feature=list(df.columns))




# Various imports
import sys
sys.path.append("./modules")
import vistk
import pandas as pd
import json
import numpy as np

# Loading metadata files
metadata = pd.read_json('sourceData/sitc_metadata_int_atlas.csv')
metadata.code = metadata.code.astype(int).astype(str).str.zfill(4)
data = pd.read_csv('sourceData/master_data.csv', 
                 usecols=["year", "iso3", "sitc4", "imports", "exports", "quality_imp", "quality_exp"]).sort(columns='year')
data.sitc4 = data.sitc4.astype(int).astype(str).str.zfill(4)
df = pd.merge(data, metadata, how='left', left_on='sitc4', right_on='code')
df['category'] = df['sitc4'].map(lambda x: str(x)[0])
df.head()

df_LKA = df[(df['iso3'] == 'LKA')]
df_LKA

linechart = vistk.Linechart(id='sitc4', x='year', y='quality_exp', color='sitc4', name='name',
                           group='category', y_invert=False)
linechart.draw(df[(df['iso3'] == 'FRA')])





country = 'LKA'
caterplotTime = vistk.Caterplot(id='sitc4', color='color', name='name', x='year', 
                            y='quality_exp', r='exports', year=year, group='category')
caterplotTime.draw(df[(df['iso3'] == country)])

country = 'LKA'
year = 1984
df_caterplot = df[(df['year'] == year) & (df['iso3'] == country)]
caterplot = vistk.Caterplot(id='sitc4', color='color', name='name', x='category', 
                            y='quality_exp', r='exports', year=year, group='category')
caterplot.draw(df_caterplot)

df_caterplot.head()

with open('/Users/rvuillemot/Dev/vis-toolkit-datasets/data/quality_%s_%s.json' % (country, year), 'w') as fp:
    fp.write(df_caterplot.to_json(orient='records'))

year = 2009
df_caterplot = df[(df['year'] == year) & (df['iso3'] == country)]
caterplot = vistk.Caterplot(id='sitc4', color='color', name='name', x='category', 
                            y='quality_exp', r='exports', year=year, group='category')
caterplot.draw(df_caterplot)

with open('/Users/rvuillemot/Dev/vis-toolkit-datasets/data/quality_%s_%s.json' % (country, year), 'w') as fp:
    fp.write(df_caterplot.to_json(orient='records'))

df_caterplot.head()

filter_country = 'South Asia'
highlight_country = 'LKA'
title = 'Quality Distribution by country through time: ' + filter_country



country = 'FRA'
year = 1984
caterplot = vistk.Caterplot(id='sitc4', color='color', name='name', x='category', 
                            y='quality_exp', r='exports', year=year, group='category')
caterplot.draw(df[(df['year'] == year) & (df['iso3'] == country)])

# Generate a pretty table
from ipy_table import *
import numpy as np

df_table = df[(df['year'] == 1984)].head(10).reset_index(drop=True).reset_index()
df_table = df_table[['sitc4', 'name']]
table = df_table.as_matrix()

header = np.asarray(df_table.columns)
header[0] = 'Code'
header[0] = 'Description'
# df.rename(columns=lambda x: x[1:], inplace=True)
table_with_header = np.concatenate(([header], table))

# Basic themes
# Detais http://nbviewer.ipython.org/github/epmoyer/ipy_table/blob/master/ipy_table-Introduction.ipynb
make_table(table_with_header)
apply_theme('basic')
# Only show the top-10
set_row_style(1, color='yellow')


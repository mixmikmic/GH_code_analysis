import pandas as pd
from psycopg2 import connect
con = connect(database='census')

# We leave out summary level 500 -- congressional districts -- because they changed considerably between
# 2007 and 2014.
sql = """
select g.name, d.* 
from {schema}.geoheader g, {schema}.b03002 d
where g.geoid = d.geoid
and g.sumlevel != 500
"""

b03002_14 = pd.read_sql_query(sql.format(schema='acs2014_1yr'), con, index_col='geoid')
b03002_07 = pd.read_sql_query(sql.format(schema='acs2007_1yr'), con, index_col='geoid')

def simplifier(b03002,percentify=True):
    """Given a dataframe with B03002 schema, reduce it to a simpler dataframe according to the rules described above"""
    newcols = {
        'total': b03002['b03002001'],
        'white': b03002['b03002003'],
        'black': b03002['b03002004'],
        'amerind': b03002['b03002005'],
        'asian': b03002['b03002006'],
        'pacific': b03002['b03002007'],
        'other': b03002['b03002008'] + b03002['b03002009'],
        'hisp': b03002['b03002012']
    }
    if percentify:
        total = newcols.pop('total')
        for col in newcols:
            newcols[col] = newcols[col]/total
            
    return pd.DataFrame.from_dict(newcols)

race07 = simplifier(b03002_07)
race14 = simplifier(b03002_14)

race_chg = race14 - race07
race_chg = race_chg.dropna()

# it's nice to have the 'name' column at the front when spitting it out in the notebook. 
# Since we use the geoid for the dataframe index, we can comfortably mix and match data from different queries.
race_chg.insert(0,'name', b03002_07['name'])

index=[]
summary = {'col':[],'value': [], 'name': []}
for col in race_chg.columns:
    if col != 'name':
        top = race_chg[col].max()
        bot = race_chg[col].min()
        if abs(top) > abs(bot):
            val = top
        else:
            val = bot
        row = race_chg[race_chg[col] == val]
        summary['col'].append(col)
        summary['value'].append(val)
        summary['name'].append(row['name'].item()) # we don't want the index label so use item()
        index.append(row.index.tolist()[0]) # here's where we want the index/geoid

summary = pd.DataFrame(summary,index=pd.Series(index,name='geoid'))
summary

race = pd.merge(race07,race14,left_index=True,right_index=True,suffixes=('_07','_14'))
race = race.merge(race_chg,left_index=True,right_index=True,suffixes=('','_chg'))
race.insert(0,'name',race.pop('name'))
race['total_07'] = b03002_07['b03002001']
race['total_14'] = b03002_14['b03002001']
race = race.merge(race_chg,left_index=True,right_index=True,suffixes=('','_chg'))
race[[x for x in race.columns if x.endswith('_chg')]].describe()

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
race_chg.plot(kind='box')

from bokeh.charts import Scatter, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import HoverTool, PolyAnnotation
from bokeh.models.sources import ColumnDataSource
output_notebook()

#scatterplot = Scatter(race, x='white_07', y='white_14', title="Pct white 2007 vs 2014",
#            xlabel="2007 pct white")
# http://blog.rtwilson.com/bokeh-plots-with-dataframe-based-tooltips/
hover = HoverTool(tooltips=[('name','@name'),
                            ('geoid', '@geoid'),
                            ('white', '@white_chg{1.11}'), 
                            ('black', '@black_chg{1.11}'),
                            ('hisp', '@hisp_chg{1.11}'),
                            ('asian', '@asian_chg{1.11}')])

p = figure(plot_width=750, plot_height=750,
           title="larger swings")
p.add_tools(hover)

_plot_configs = [
    ('white', 'red', 'circle'),
    ('black', 'blue', 'square'),
    ('hisp', 'green', 'triangle'),
    ('asian', 'brown', 'inverted_triangle')
]

for r, c, m in _plot_configs:
    x_col = (r+'_07')
    y_col = (r+'_14')
    chg_col = (r+'_chg')
    race_for_plot = race[(race[chg_col].abs() > 0.2)]
    p.scatter(x=x_col, y=y_col, size=10, color=c, marker=m, source=ColumnDataSource(race_for_plot), legend=chg_col)


p.xaxis.axis_label = '2007 pct of population'
p.yaxis.axis_label = '2014 pct'

# these just don't work!
# drop_anno = PolyAnnotation(plot=p, xs=[.2, .4, .4, .2], ys=[.2, .2, .4, .4], fill_alpha=0.1, fill_color='red')
#growth_anno = PolyAnnotation(plot=p, bottom=180, fill_alpha=0.1, fill_color='green')

#p.renderers.extend([drop_anno, ])

show(p)

race[race['black_chg'].abs() > 0.2][['name','black_07','black_14','black_chg','total_07', 'total_14']]

race[race['asian_chg'].abs() > 0.2][['name','asian_07','asian_14','asian_chg','total_07', 'total_14']]

race[race['hisp_chg'].abs() > 0.2][['name','hisp_07','hisp_14','hisp_chg','total_07', 'total_14']]

race[race['white_chg'].abs() > 0.2][['name','white_07','white_14','white_chg','total_07', 'total_14']]


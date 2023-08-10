get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
from bokeh.io import output_notebook
from bokeh.plotting import figure, show


from bokeh.io import show, output_notebook
from bokeh.models import (
    GeoJSONDataSource,
    ColumnDataSource,
    HoverTool,
    LogColorMapper
)
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure

import json
from collections import OrderedDict
output_notebook()

import cbsodata

df_income = pd.DataFrame(cbsodata.get_data('80592NED'))
# df_income = pd.read_pickle('df_income.pickle')  # 



# Take a look at the data from CBS:
df_income.head(15)

df_income.groupby('HoogteVanHetInkomen').size()

df_income['GemiddeldBesteedbaarInkomen_3'].plot(kind = 'hist')
plt.show()

# Only keep the income data from 2014 (the most recent). 
df_income = df_income[df_income['Perioden'] == '2014']

# CBS does something with upper quartile of local population and lower and etc. and etc. 
# We don't want all that, only keep full regional population
df_income = df_income[df_income['HoogteVanHetInkomen'] == 'Totaal huishoudens']

# Finally, there is an overload in columns we don't really need. Only keep the columns with region and average income.
df_income = df_income[['RegioS', 'GemiddeldBesteedbaarInkomen_3']]

# Decimals are weird from cbs for some reason
df_income['GemiddeldBesteedbaarInkomen_3'] = np.around(df_income['GemiddeldBesteedbaarInkomen_3'], decimals=0)

df_income = df_income[df_income['GemiddeldBesteedbaarInkomen_3'].notnull()]

# Set index for later purposes
df_income = df_income.set_index('RegioS')


# read all municipalities
# SOURCE: https://www.webuildinternet.com/2015/07/09/geojson-data-of-the-netherlands/

# os.chdir(r'LOCATION\OF\YOUR\FILE')

with open(r'Gemeenten.geojson', 'r') as f:
    dutch_municipalities_dict = json.loads(f.read(), object_hook=OrderedDict)

# See how this geojson looks. It's basically a dictionary with many polygons, 
# which dictate the outer edges of (in this case) the municipalities. 
# Also, there is an 'properties' part connected to each set of locations, which holds additional information. 
# Already included are the municipality names, code and some more features. 
# We will add one more feature ourselves here: the average regional income. 

# Can't run this in Github since it displays the entire dictionary which is very large. 
# dutch_municipalities_dict

# Using a screenshot instead.
# from IPython.display import Image
# Image("GEOJSON_1.png")

dutch_municipalities_dict

# Some names we can fix, some not.
# You should really do this with a dictionary or something but at least this works

# This is an iterative process between this and the list 'unfindable' as defined below.
# This code needs to be run first though. 

df_income['GemiddeldBesteedbaarInkomen_3']['Dantumadeel (Dantumadiel)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Dantumadiel']
df_income['GemiddeldBesteedbaarInkomen_3']['Ferwerderadeel (Ferwerderadiel)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Ferwerderadiel']
df_income['GemiddeldBesteedbaarInkomen_3']['Littenseradeel (Littenseradiel)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Littenseradiel']
df_income['GemiddeldBesteedbaarInkomen_3']['Menaldumadeel (Menaldumadiel)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Menameradiel']
df_income['GemiddeldBesteedbaarInkomen_3']['SÃºdwest-FryslÃ¢n'] = df_income['GemiddeldBesteedbaarInkomen_3']['Súdwest-Fryslân']
df_income['GemiddeldBesteedbaarInkomen_3']['Tietjerksteradeel (Tytsjerksteradiel)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Tytsjerksteradiel']


df_income['GemiddeldBesteedbaarInkomen_3']['Bergen (L)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Bergen (L.)']
df_income['GemiddeldBesteedbaarInkomen_3']['Bergen (NH)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Bergen (NH.)']
df_income['GemiddeldBesteedbaarInkomen_3']['Utrecht (Ut)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Utrecht (gemeente)']
df_income['GemiddeldBesteedbaarInkomen_3']['Groningen (Gr)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Groningen (gemeente)']
df_income['GemiddeldBesteedbaarInkomen_3']['Beek'] = df_income['GemiddeldBesteedbaarInkomen_3']['Beek (L.)']
df_income['GemiddeldBesteedbaarInkomen_3']['Stein'] = df_income['GemiddeldBesteedbaarInkomen_3']['Stein (L.)']
df_income['GemiddeldBesteedbaarInkomen_3']['Hengelo (O)'] = df_income['GemiddeldBesteedbaarInkomen_3']['Hengelo (O.)']
df_income['GemiddeldBesteedbaarInkomen_3']['Middelburg'] = df_income['GemiddeldBesteedbaarInkomen_3']['Middelburg (Z.)']
df_income['GemiddeldBesteedbaarInkomen_3']['Rijswijk'] = df_income['GemiddeldBesteedbaarInkomen_3']['Rijswijk (ZH.)']
df_income['GemiddeldBesteedbaarInkomen_3']["'s-Gravenhage"] = df_income['GemiddeldBesteedbaarInkomen_3']["'s-Gravenhage (gemeente)"]
df_income['GemiddeldBesteedbaarInkomen_3']['Laren'] = df_income['GemiddeldBesteedbaarInkomen_3']['Laren (NH.)']

# Function below will add in the cbs data we have. 
# This is based on crossing the names: it will look for the name from the geojson in the cbs data
# This will work for many municipalities, but there will unfortunately be cases in which the municipality can't be found. 

# Python will give a KeyError in those cases. Because of changes in municipalities (Gemeentelijke herindelingen) 
# we won't be able to fix all name incompatabilities, but many we will. Therefore, store them in  a separate file. 

unfindable = []

def merge_income(dutch_municipalities_dict, df_income):
    
    municipality = dutch_municipalities_dict['properties']['name']
    
    try: 
        dutch_municipalities_dict['properties']['AverageIncome'] = round(df_income['GemiddeldBesteedbaarInkomen_3'][municipality],0).astype('float')
    except KeyError:
        unfindable.append(municipality)
        dutch_municipalities_dict['properties']['AverageIncome'] = 30.00

    return dutch_municipalities_dict

# merge income: execute the function here. 
dutch_municipalities_dict['features'] = [merge_income(municipality, df_income) for municipality in dutch_municipalities_dict['features']]

unfindable

# See how the GEOJSON has changed since the previous one: a new property -AverageIncome - is added
# Image("GEOJSON_2.png")
dutch_municipalities_dict

# Maasdonk is the only municipality we are unable to fix
# Some further research (i.e. Wikipedia) learns that this municipality has been assimilated during replanning. 

# Bokeh requires for all municipalities to have a value for AverageIncome. 
# We can't give this 0, since that would impact the map itself too much (it needs to have a color for the entire range)
# Therefore, we leave in the function that this municipality gets a value for AverageIncome of 30.0

df_income[df_income.index.str.contains('Maasdonk')]

geo_source = GeoJSONDataSource(geojson=json.dumps(dutch_municipalities_dict))

palette.reverse()

color_mapper = LogColorMapper(palette=palette)

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

p = figure(
    title="Inkomen per gemeente", tools=TOOLS,
    x_axis_location=None, y_axis_location=None
)
p.grid.grid_line_color = None

p.patches('xs', 'ys', source=geo_source,
          fill_color={'field': 'AverageIncome', 'transform': color_mapper},
          fill_alpha=0.7, line_color="white", line_width=0.5)

hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [
    ("Name", "@name"),
    ("Code", "@code"),
    ("AverageIncome", "@AverageIncome"),
    ("Level", "@level"),
    ("(Long, Lat)", "($x, $y)"),
]

show(p)

# The image should look like below (colors may vary sometimes)
Image("Bokeh_map.png")




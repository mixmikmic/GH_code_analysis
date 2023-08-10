get_ipython().system('pip install pandas')
get_ipython().system('pip install requests')
get_ipython().system('pip install cenpy')
get_ipython().system('pip install pysal')

import pandas as pd
import cenpy as cen
import pysal

datasets = list(cen.explorer.available(verbose=True).items())

# print first rows of the dataframe containing datasets
pd.DataFrame(datasets).head()

dataset = '2012acs1'
cen.explorer.explain(dataset)

con = cen.base.Connection(dataset)
con

print(type(con))
print(type(con.geographies))
print(con.geographies.keys())

# print head of data frame in the geographies dictionary
con.geographies['fips'].head()

g_unit = 'county:*'
g_filter = {'state':'8'}

var = con.variables
print('Number of variables in', dataset, ':', len(var))
con.variables.head()

cols = con.varslike('B01001A_')
cols.extend(['NAME', 'GEOID'])

data = con.query(cols, geo_unit=g_unit, geo_filter=g_filter)
# prints a deprecation warning because of how cenpy calls pandas

data.index = data.NAME

# print first five rows and last five columns
data.ix[:5, -5:]

cen.tiger.available()

con.set_mapservice('tigerWMS_ACS2013')

# print layers
con.mapservice.layers

geodata = con.mapservice.query(layer=84, where='STATE=8')

# preview geodata
geodata.ix[:5, :5]

newdata = pd.merge(data, geodata, left_on='county', right_on='COUNTY')
newdata.ix[:5, -5:]




import pandas as pd

import matplotlib.pyplot as plt

#When this cell is run, a simple line chart should be displayed
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()

from sqlalchemy import create_engine
engine = create_engine("postgresql://test:test@postgres:5432/tm351test")

#Run a simple query on a default table
from pandas import read_sql_query as psql

psql("SELECT table_schema,table_name FROM information_schema.tables     ORDER BY table_schema,table_name LIMIT 3;", engine)
#A table containing three rows should appear

#Load in the sql extensions - I wonder if we should try to autoload this?
get_ipython().magic('load_ext sql')
#This is how we connect to a sql database
#Monolithic VM addressing style
get_ipython().magic('sql postgresql://tm351admin:tm351admin@postgres:5432/tm351test')

get_ipython().run_cell_magic('sql', '', 'SELECT CURRENT_USER;')

get_ipython().magic('load_ext sql')
get_ipython().magic('sql postgresql://test:test@postgres:5432/tm351test')

get_ipython().run_cell_magic('sql', '', 'SELECT table_schema,table_name FROM information_schema.tables ORDER BY table_schema,table_name LIMIT 1;')

demo = get_ipython().magic('sql SELECT table_schema FROM information_schema.tables LIMIT 3')
demo

import pymongo
from pymongo import MongoClient

#If connecting to the default port, you can omit the second (port number) parameter
# Open a connection to the Mongo server, open the accidents database and name the collections of accidents and labels
c = MongoClient('mongodb', 27017)

c.database_names()

import seaborn

from numpy.random import randn
data = randn(75)
plt.hist(data)
#Running this cell should produce a histogram.

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
#Running this cell should produce a line chart.

import folium
#Note - this will not display a map if your are offline.

#A network connection is required to retrieve the map tiles
osmap = folium.Map(location=[52.01, -0.71], zoom_start=13,height=500,width=800)
osmap.simple_marker([52.0250, -0.7056], popup='The <b>Open University</b> campus.')
osmap.render_iframe = True
osmap.create_map()
osmap




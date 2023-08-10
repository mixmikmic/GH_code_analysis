import pandas as pd

import matplotlib.pyplot as plt

#When this cell is run, a simple line chart should be displayed
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()

from sqlalchemy import create_engine
engine = create_engine("postgresql://test:test@localhost:5432/tm351test")

#Run a simple query on a default table
from pandas import read_sql_query as psql

psql("SELECT table_schema,table_name FROM information_schema.tables     ORDER BY table_schema,table_name LIMIT 3;", engine)
#A table containing three rows should appear

get_ipython().magic('load_ext sql')
get_ipython().magic('sql postgresql://test:test@127.0.0.1:5432/tm351test')

get_ipython().run_cell_magic('sql', '', 'SELECT table_schema,table_name FROM information_schema.tables ORDER BY table_schema,table_name LIMIT 1;')

demo = get_ipython().magic('sql SELECT table_schema FROM information_schema.tables LIMIT 3')
demo

import pymongo
from pymongo import MongoClient

#If connecting to the default port, you can omit the second (port number) parameter
# Open a connection to the Mongo server, open the accidents database and name the collections of accidents and labels
c = pymongo.MongoClient('mongodb://localhost:27351/')

c.database_names()

db = c.accidents
accidents = db.accidents
accidents.find_one()

#Quick way to kill all mongo processes...
#!killall mongod
#!killall mongos
#...then bring the base mongo server as service on 27351 back up
#!systemctl restart mongodb

get_ipython().system('/etc/mongo-shards-down')
get_ipython().system('/etc/mongo-shards-up')

c2 = pymongo.MongoClient('mongodb://localhost:27017/')
c2.database_names()

#Test a query on the sharded database
db = c2.accidents
accidents = db.accidents
accidents.find_one()

#Turn the sharded server off
get_ipython().system('/etc/mongo-shards-down')

import seaborn

from numpy.random import randn
data = randn(75)
plt.hist(data)
#Running this cell should produce a histogram.

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot();
#Running this cell should produce a line chart.

import folium
#Note - this will not display a map if you are offline.

#A network connection is required to retrieve the map tiles
osmap = folium.Map(location=[52.01, -0.71], zoom_start=13,height=500,width=800)
folium.Marker([52.0250, -0.7056], popup='The <b>Open University</b> campus.').add_to(osmap)
osmap.render_iframe = True
osmap.save('test.html')
osmap




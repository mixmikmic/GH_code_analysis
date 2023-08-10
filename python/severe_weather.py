# import necessary libraries
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import inspect, create_engine, func, MetaData, Table
import datetime as dt
from datetime import time
import pandas as pd
import numpy as np
import json

# Database Setup
engine = create_engine('sqlite:///SevereWeather.sqlite', echo=False)

# produce our own MetaData object
metadata = MetaData()

# we can reflect it ourselves from a database, using options
# such as 'only' to limit what tables we look at...
metadata.reflect(engine)

Base = automap_base()
Base.prepare(engine, reflect=True)

inspector = inspect(engine)
Events = Table('Events',metadata)
inspector.reflecttable(Events, None)
session = Session(bind=engine)

events_df = pd.read_sql_table('Events', engine)

events_df.info()

# Make the date_time a string
events_df['date_time'] = [dt.date.strftime(x,'%Y-%m-%d : %H:%M:%S') for x in events_df.date_time]

events_df.info()

tornado_df = events_df.loc[events_df['type']=="torn"]

# Convert the df to a csv file. 
# Upload the csv file into a geojson converter.
tornado_df.to_csv('tornado.csv', encoding='utf-8', index=False)

hail_df = events_df.loc[events_df['type']=="hail"]

# Convert the df to a csv file. 
# Upload the csv file into a geojson converter.
hail_df.to_csv('hail.csv', encoding='utf-8', index=False)

wind_df = events_df.loc[events_df['type']=="wind"]

# Convert the df to a csv file. 
# Upload the csv file into a geojson converter.
wind_df.to_csv('wind.csv', encoding='utf-8', index=False)




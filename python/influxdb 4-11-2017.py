"""Tutorial for using pandas and the InfluxDB client."""

import argparse
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from influxdb import DataFrameClient

#host='localhost'
host='192.168.0.30'
port=8086   # port of InfluxDB http API

"""Instantiate the connection to the InfluxDB client."""
user = 'root'
password = 'root'
dbname = 'home_assistant'

client = DataFrameClient(host, port, user, password, dbname)

fields = """ * """
measurement = """ "°C" """
filters = """WHERE "value" < 18.5 """

# query = """SELECT * FROM "°C" WHERE "value" < 18.5"""
query = """SELECT {} FROM {} {} """.format(fields, measurement, filters)
print("Performing a query: {}".format(query))

response = client.query(query)

print(response)

print(response.keys())

response_df = pd.DataFrame.from_dict(response['°C'])

response_df

response_df.columns

response_df[response_df['friendly_name_str'] == 'Mean temperature']['mean'].plot()

fields = """ * """
measurement = """ "%" """
filters = """ """
query = """SELECT {} FROM {} {} """.format(fields, measurement, filters)
print("Performing a query: {}".format(query))

response = client.query(query)

print(response.keys())

response_df = pd.DataFrame.from_dict(response['%'])

response_df.head()

response_df['entity_id'].unique()

bme680air_qual_df = response_df[response_df['entity_id'] == 'bme680air_qual']

bme680air_qual_df.tail()

bme680air_qual_df['value'].plot(figsize=(20,3))

bme680humidity_df = response_df[response_df['entity_id'] == 'bme680humidity']

bme680humidity_df['value'].plot(figsize=(20,3))




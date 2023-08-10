import pandas as pd
from pandas import DataFrame
import sqlalchemy

weather_data = pd.read_csv('2015.csv', 
                           header=None,
                           index_col=False,
                           names=['station_identifier', 
                                  'measurement_date', 
                                  'measurement_type', 
                                  'measurement_flag', 
                                  'quality_flag', 
                                  'source_flag', 
                                  'observation_time'],
                           parse_dates=['measurement_date'])

weather_data.head()

weather_data_subset = weather_data[weather_data.measurement_type.isin(['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'])][['station_identifier', 'measurement_date', 'measurement_type', 'measurement_flag']]

weather_data_subset.head()

db_name = 'weather'
connection_string = "postgresql://localhost:5432/%s" % (db_name)
conn = sqlalchemy.create_engine(connection_string)

table_name = 'weather_data'
# The to_sql method defaults to bigint for integer types here, which are larger than we need. 
# This manually sets the datatypes of the columns we need to override
column_type_dict = {'measurement_flag': sqlalchemy.types.Integer}
# Writing all the data to the DB at once will cause this notebook to crash.
# We pass a large integer to the chunksize parameter to chunk the writing of records
weather_data_subset.to_sql(table_name, conn, chunksize=100000, index_label='id', dtype=column_type_dict)

station_metadata = pd.read_csv('ghcnd-stations.txt', 
                           sep='\s+',  # Fields are separated by one or more spaces
                           usecols=[0, 1, 2, 3],  # Grab only the first 4 columns
                           na_values=[-999.9],  # Missing elevation is noted as -999.9
                           header=None,
                           names=['station_id', 'latitude', 'longitude', 'elevation'])

station_metadata.head()

len(station_metadata[station_metadata['elevation'].isnull()])

metadata_table_name = 'station_metadata'
station_metadata.to_sql(metadata_table_name, conn, index_label='id')

weather_type_dict = {'PRCP': 'Precipitation', 'SNOW': 'Snowfall', 'SNWD': 'Snow Depth', 
                     'TMAX': 'Maximum temperature', 'TMIN': 'Minimum temperature'}
weather_type_df = DataFrame(weather_type_dict.items(), columns=['weather_type', 'weather_description'])
description_table_name = 'weather_types'
weather_type_df.to_sql(description_table_name, conn, index_label='id')


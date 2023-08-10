from pandas import read_csv
import pandas as pd
from datetime import datetime
from geopy.distance import great_circle

def clean_html(df):
    df.replace({r'<.*?>': ''}, regex=True, inplace=True)

# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('HurricaneData/PRSA_data_2010.1.1-2014.12.31.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('HurricaneData/Preprocessed/pollution.csv')

def parse(x):
	return datetime.strptime(x, '%Y-%m-%d')
dataset = read_csv('HurricaneData/houston.csv', index_col=0)
print("Original dataset:")
print(dataset.head(5))
dataset.set_index(dataset['Date'], inplace=True)
dataset.index = pd.to_datetime(dataset.index)
dataset.drop(dataset.columns[0], axis=1, inplace=True)

# mark all NA values with 0
dataset.fillna(0, inplace=True)

clean_html(dataset)

# summarize first 5 rows
print("Processed dataset:")
print(dataset.head(5))
# save to file
dataset.to_csv('HurricaneData/Preprocessed/houston.csv')

houston_lat_lon = (29.76043, -95.36980)

def location_filter(x):
    distance = great_circle((x['Latitude'], x['Longitude']), houston_lat_lon).miles
    return distance < 700
    
def parse_date(x):
    split = x.split(" ")
    return datetime.strptime("%s %s" % (split[0], split[1].zfill(4)), '%Y%m%d %H%M')
    
dataset = read_csv('HurricaneData/atlantic.csv', parse_dates=[['Date', 'Time']], date_parser=parse_date, index_col=0)
print("Original dataset:")
print(dataset.head(5))
dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset.index.name = 'Date'
dataset.index = pd.to_datetime(dataset.index)

dataset['Latitude'].replace({r'[^0-9]': ''}, regex=True, inplace=True)
dataset['Longitude'].replace({r'[^0-9]': ''}, regex=True, inplace=True)
dataset['Latitude'] = pd.to_numeric(dataset['Latitude'], downcast='float')
dataset['Longitude'] = pd.to_numeric(dataset['Longitude'], downcast='float')
dataset = dataset[dataset.apply(location_filter, axis=1)]
dataset.replace(-999, 0, inplace=True)
# summarize first 5 rows
print("Processed dataset:")
print(dataset.head(5))
# save to file
dataset.to_csv('HurricaneData/Preprocessed/hurdat_houston.csv')




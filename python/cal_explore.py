import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')
get_ipython().magic('run Pipeline//classify_and_evaluate')
get_ipython().magic('run Pipeline//upload_and_vizualize')
get_ipython().magic('run Pipeline//aux')
import pylab as pl
import seaborn as sn
from datetime import datetime as dt
from datetime import date 
import geopandas as gpd

cal = pd.read_csv('data/calendar_chicago_2015.csv')
cal_copy = cal.copy()

cal.head()

cal.transpose().head()

cal.dtypes

dprint(dt.strptime(cal.loc[1].date,'%Y-%m-%d'))

def convert_to_datetime(series_row):
    return dt.strptime(series_row,'%Y-%m-%d')

def convert_to_weekday(series_row,output):
    if output == 'day_num':
        return date.weekday(series_row)
    output_dict = {"weekday":'%A', "month_name":'%B',"month_num":'%m'}
    return date.strftime(series_row,output_dict[output])

def convert_to_bool(df, column, conversion):
    return df[column].replace(conversion)

def add_date_cols(df, date_column):
    df['datetime'] = df[date_column].apply(convert_to_datetime)
    df['day_num'] = df.datetime.apply(convert_to_weekday, output='day_num')
    df['day_of_week'] = df.datetime.apply(convert_to_weekday, output='weekday')
    df['month'] =  df.datetime.apply(convert_to_weekday, output='month_name')
    df['month_num'] =  df.datetime.apply(convert_to_weekday, output='month_num')

def get_occupied_frame(df, occ_column, date_column, conversion, bool_param):
    add_date_cols(df,date_column)
    df[occ_column] = convert_to_bool(df,occ_column,conversion)
    return df[df[occ_column] == bool_param]

d = {'t': True, 'f': False}
cal_occupied = get_occupied_frame(cal,'available','date', d, True)

cal_occupied.count()

cal_occupied.dtypes


cal_unavailable_monthXDay = cal_occupied.sort(['month_num','day_num']).groupby(['month','day_of_week'], sort=False).size().to_frame()

cal_unavailable_monthXDay.plot(kind='line',figsize=(30,40), fontsize=18,rot=70)




fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name,i in cal_occupied.sort('month_num').groupby(['listing_id','month'],sort=False):#.available.count().to_frame():#.plot(kind='scatter')
    fig, ax = plt.subplots()
    print(name,i)
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    ax.plot(i[0], i[1], marker='o', linestyle='', ms=12)

small = chicago_listing[['id','price']]

def convert_to_float(series_row):
    return float(series_row.replace('$','').replace(',',''))

#small['priceNo$'] = small.price.apply(convert_to_float)
small.head()

listing = []
avg_price = []
for index, i in enumerate(small.groupby('id').mean().iterrows()):#.available.count().to_frame():#.plot(kind='scatter')
    listing.append(i[0])
    avg_price.append(i[1][0])

fig = plt.figure(figsize=(20,20))
plt.xticks(listing, listing, rotation=80)
ax = fig.add_subplot(1,1,1)
#ax.xticks(listing,None,rotation='vertical')
ax.set_ylabel('Average Price per Listing', fontsize=20)
ax.set_xlabel('Listing_Id', fontsize=12)
for index, i in enumerate(small.groupby('id').mean().iterrows()):#.available.count().to_frame():#.plot(kind='scatter')
    ax.plot(i[0], i[1], marker='o', linestyle='', ms=12)

small.groupby('id').mean().hist(bins=100, figsize=(20,20))
plt.xticks(range(0,1500,50), rotation=70)
plt.xlim(0,1500,25)

chicago_listing = read_file('data/chicago_listings.csv')

all_cols, all_cols_caps, pt_columns = list_describe(chicago_listing)

create_hist_box(chicago_listing,all_cols,['id','name','host_name','neighbourhood_group','neighbourhood','room_type','latitude','longitude','room_type','last_review'])

chicago_listing.head(2)

n_hoods = [(row[0]) for row in chicago_listing.neighbourhood.value_counts().to_frame().iterrows()]

fig, ax = plt.subplots(1,1) 

chicago_listing.neighbourhood.value_counts().plot(kind='bar', figsize=(15,15))

ax.set_xticklabels(n_hoods,rotation='vertical')
plt.show()

neighborhoods = chicago_listing[['neighbourhood','price']]
neighborhoods.groupby('neighbourhood').mean().plot(kind='bar', figsize=(20,20),color = 'green')
#plt.xticks(range(0,1500,50), rotation=70)
#plt.xlim(0,1500,25)

fname = 'data/Boundaries - ZIP Codes.geojson'
chicago = gpd.read_file(fname)
chicago.head()

base = chicago.plot(column='shape_area', colormap='YlGn', figsize=(30,30))
chicago_listing.plot(ax=base,kind='scatter', x='longitude', y='latitude', figsize=(30,30), s=9, c='red')

check_na(chicago_listing)




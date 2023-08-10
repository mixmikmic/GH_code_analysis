from format_train_test import *
import matplotlib.pyplot as plt
from pytz import timezone
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

filename = '../data/NDBC_all_data_all_years.csv'
buoyID_train = [46059]
buoyID_test = [46026]

data_train_46059 = get_train_bouys(filename, buoyID_train[0])
data_labels_46026  = get_train_bouys(filename, buoyID_test[0])

yr_lst = [1994,1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
              2003, 2004, 2006, 2007, 2008, 2012]
data_train_46059_hr = join_all_hourly_data(data_train_46059, yr_lst)
data_labels_46026_hr  = join_all_hourly_data(data_labels_46026, yr_lst)

data_train_46059_t = adding_speed_col(data_train_46059_hr , 650)
data_train_46059_t = add_time_delta(data_train_46059_t)
data_train_46059_t = add_time_y(data_train_46059_t)
data_train_46059_t = round_time_y(data_train_46059_t)
data_X_y_46059 = pd.merge(data_train_46059_t,
                          data_labels_46026_hr,
                          how='left', left_on='time_y_hr', right_on='id')

data_X_y_46059.tail(50)

data_labels_46026_hr.head()

data_train_46059_hr.head()

fig, ax = plt.subplots(figsize=(18,6))
data_train_46059_hr['2008']['WVHT'].plot(marker ='o', 
                                         linestyle='None', 
                                         ms=10, 
                                         markerfacecolor='None', 
                                         c='r', alpha=0.5)
data_train_46059['2008']['WVHT'].plot(marker ='.', linestyle='None', alpha=0.8, c ='b')

import os
import requests
import time
from bs4 import BeautifulSoup

#a = 'https://tidesandcurrents.noaa.gov/api/datagetter?'
#b = 'begin_date=20180101 00:00&end_date=20180101 00:00&'
#c = 'station=9414290&product=hourly_height&datum=mllw&units=metric&time_zone=lst&application=ports_screen&format=json' 

a = "https://tidesandcurrents.noaa.gov/api/"
b = "datagetter?begin_date=19950101 00:00&end_date=19951231 23:00&"
c = "station=9414290&product=air_temperature&datum=mllw&units=metric&"
d = "time_zone=gmt&application=web_services&format=json "
url = a + b + c + d
data = requests.get(url)
url

a = "https://tidesandcurrents.noaa.gov/api/datagetter?product=air_pressure&"
b = "application=NOS.COOPS.TAC.MET&begin_date=20000101&end_date=20001231&"
c = "station=9414290&time_zone=lst_ldt&units=metric&interval=h&format=json"
urlpres = a+b+c
presdata = requests.get(urltemp)

presdata.json()



data

testdf = pd.DataFrame(data['data'])
testdf.drop(columns=['f','s'], inplace=True)
cols = ['Date', 'WaterLevel']
testdf.columns = cols
testdf['WaterLevel'] = testdf['WaterLevel'].apply(lambda x: float(x))
testdf.head()



import wget

filename = wget(url)

test = pd.read_csv('data_X_y_46059_train_012918.csv', parse_dates=['id_x'], index_col='id_x')

test.head().T














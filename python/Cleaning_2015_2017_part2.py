# Imports
import pandas as pd
import numpy as np

# Load Data
weather_2015_2017 = pd.read_csv('cleaned_weather_2015_2017.csv', index_col=0)

weather_2015_2017.head()

len(weather_2015_2017)

beijing_spatial_15_17 = pd.read_csv('merged_spatial_weather_2015_2017.csv', index_col=0)

beijing_spatial_15_17.head()

weather_2015_2017['pollution'] = beijing_spatial_15_17['avg_air_pollution']

weather_2015_2017.head()

UCI_2010_2014 = pd.read_csv('pollution.csv', index_col=0).reset_index(drop=True)

UCI_2010_2014.head()

merged_final_pollution = weather_2015_2017.copy()

merged_final_pollution.columns = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain',
                                 'year', 'month', 'day', 'hour', 'pollution']

merged_final_pollution.head()

merged_final_pollution.to_csv('UCI_year_mo_day_pollution_weather_2015_2017.csv')

UCI_2010_2014.head()

UCI_2015_2017 = merged_final_pollution.copy()

UCI_2015_2017 = UCI_2015_2017.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=False)

UCI_2015_2017 = UCI_2015_2017[['pollution','dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']]

def preprocess_parsed_col(df, column='wnd_dir'):
    '''
    Redo parsing for wind direction
    '''
    df[column] = df[column].apply(lambda x: wind_categories(x))
    return df

def wind_categories(x):
    x = int(x)
    if x >= 0 and x <= 90:
        # Angular degrees from True north
        y = 'NE'
    if x > 90 and x <= 180:
        y = 'SE'
    if x > 180 and x <=270:
        y = 'SW'
    if x > 270 and x <=360:
        y = 'NW'
    return y

UCI_2015_2017 = preprocess_parsed_col(UCI_2015_2017)

merged_final_pollution.head()

UCI_2015_2017.head()

UCI_2010_2017 = pd.concat((UCI_2010_2014, UCI_2015_2017), axis=0)

UCI_2015_2017.to_csv('merged_final_UCI_format.csv')

UCI_2010_2017 = pd.read_csv('merged_final_UCI_format.csv', index_col=0)

def cast_float_col(df, column='dew'):
    '''
    Redo parsing for dew
    '''
    df[column] = df[column].apply(lambda x: float(x))
    return df

def fix_snow_values(df, column='snow'):
    df[column] = df[column].apply(lambda x: 0 if x == ' ' else x)
    df[column] = df[column].apply(lambda x: 0 if x in ['O','R','E'] else x)
    df[column] = df[column].apply(lambda x: 0 if type(x) != str else int(x))
    return df

def what_type(df, col='snow'):
    return df[col].apply(lambda x: type(x))

snow_fixed_2010_2017 = fix_snow_values(UCI_2010_2017, column='snow')

cast_float_col(snow_fixed_2010_2017, column='snow').head()

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import time

UCI_2015_2017.to_csv('merged_final_UCI_format.csv')

data = pd.read_csv('merged_final_UCI_format.csv', index_col=0)




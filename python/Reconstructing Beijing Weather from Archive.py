# All 545110-... is weather from Beijing Airport

import pandas as pd
import numpy as np

def get_weather(file):
    with open(file,'r') as fin:
        lines = fin.readlines()
    return lines

f15 = get_weather('545110-99999-2015')

# Control Data Section
f15[0][0:60]

# Mandatory Data Section
f15[0][59:106]

len(f15)

24*265

f15[0][15:28]

f16 = get_weather('545110-99999-2016')
f17 = get_weather('545110-99999-2017')

def extract_data(line):
    # Remove ending \n
    line = line.strip()
    ## Control Data Section
    # Pos. 1-4 - Total varchar
    total_chars = line[:4]
    # Pos 5-10 - USAF weather station
    # 11-15 - WBAN id
    # 16 - 23 date
    date = line[15:23]
    # time/hour
    time_ = line[23:27]
    #### Mandatory Data Section
    ## Dew point 
    # The temperature to which a given parcel of air 
    # must be cooled at constant pressure and water vapor content in order for saturation to occur.
    dew_point = line[93:98]
    # temperature in degrees Celcius
    air_temp = line[87:92]
    # Air Pressure
    # The air pressure relative to Mean Sea Level (MSL).
    # Units - Hectopascals
    
    air_pressure = line[99:104]
    if int(air_pressure) == 99999:
        air_pressure = 'NaN'
    ## Combined wind direction
    # The angle, measured in a clockwise direction, between true north and 
    # the direction from which the wind is blowing.
    wind_dir = line[60:63]
    if int(wind_dir) == 999:
        wind_dir = 'NaN'
    ## Cumulated wind speed
    # The rate of horizontal travel of air past a fixed point
    wind_speed = line[65:69]
    
    ###### Additional Data Section
    chars_add = line[105:108]
    num_rain_attrs = line[108:111]
    # Cumulated hours of snow
    try:
        variable_index_snow = line.index('L')
        num_hours_snow = line[variable_index_snow + 2: variable_index_snow + 4]
    except ValueError as e:
        num_hours_snow = 'NaN'
        
    # Cumulated hours of rain
    try:
        var_index_additional_sec = line.index('ADD')
        num_hours_rain = int(line[var_index_additional_sec + 6: var_index_additional_sec + 8])
    except ValueError as e:
        num_hours_rain = 'NaN'
    if num_hours_rain == 99:
        num_hours_rain = 'NaN'
        
    #print(line)
    return tuple((date, time_, dew_point, air_temp, air_pressure, wind_dir, wind_speed, 
                 num_hours_snow, num_hours_rain))

def prepare_df(lines):
    return [extract_data(line) for line in lines]

def make_df(first_year, additional_years):
    all_years = first_year
    for year in additional_years:
        all_years += year
    all_years = prepare_df(all_years)
    columns = ['date', 'time', 'dew_point', 'air_temp', 'air_pressure','wind_dir', 'wind_speed',
              'cumulative_snow_hours', 'cumulative_rain_hours']
    return pd.DataFrame(all_years, columns=columns)

extract_data(f15[0])

df = make_df(f15, [f16, f17])

df['year'] = df['date'].apply(lambda x: str(x)[:4])
df['month'] = df['date'].apply(lambda x: int(str(x)[4:6]))
df['day'] = df['date'].apply(lambda x: int(str(x)[6:8]))
df['hour'] = df['time'].apply(lambda x: int(str(x)[:2]))
df['datetime'] = df['date'] + df['time']

df_unique_datetime = df.drop_duplicates(subset='datetime')

df_unique_datetime['datehour'] = df_unique_datetime['date'] +    df_unique_datetime['hour'].apply(lambda x: str(x))

df_unique_hour = df_unique_datetime.drop_duplicates(subset='datehour')

final = df_unique_hour.copy()

final = final.drop(['date', 'time', 'datetime', 'datehour'], axis=1, inplace=False)

final

final.to_csv('beijing_weather_2015_2017.csv')


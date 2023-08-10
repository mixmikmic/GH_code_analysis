import pandas as pd
import numpy as np

merged_final = pd.read_csv('./merged_final_single_station.csv', low_memory=False, index_col=0)

pollution_2010_2014 = pd.read_csv('./pollution.csv', index_col=0)

extra_weather = pd.read_csv('beijing_weather_2015_2017.csv', index_col=0).reset_index(drop=True)

extra_weather = fill_blank_spaces(extra_weather)

def fill_in_median_val(x, column_name, measure_by_class):
    '''Fill in a column name using the class, sex and title measures.'''
    if np.isnan(x[column_name]):
        x[column_name] = measure_by_class[x['year']]
    
    return x

def preprocess_air_pressure(df, grouped_class, column='air_pressure'):
    '''
    Fill nans with medians in air pressure, clean other data
    Median is grouped by year
    '''
    median_pressure_by_class = df.groupby([grouped_class])[column].median()
    df = df.apply(lambda x: fill_in_median_val(x, column, median_pressure_by_class), axis=1)
    return df

def fill_wind_dir_nans(df, grouped_class='year', column='wind_dir'):
    '''
    Fill nans in wind direction with median
        where the medians are taken from data grouped by year
    '''
    median_wind_dir_by_class = df.groupby([grouped_class])[column].median()
    df = df.apply(lambda x: fill_in_median_val(x, column, median_wind_dir_by_class),
                     axis=1)
    return df

def fix_snow_values(x):
    ''' 
    Mapping function that fixes snow values
    '''
    try:
        y = int(x)
    except ValueError as e:
        y = str(x)[1:]
    return y

def fill_snow_hours_nans(df, column='cumulative_snow_hours'):
    ''' 
    First, fill NaNs with zeroes. Then expand out the hours
    '''
    # fill columns
    df[column] = df[column].fillna(0)
    return df

def fill_rain_hours_nans(df, column='cumulative_rain_hours'):
    '''
    Fill rain hour NaNs with zeros.
    '''
    df[column] = df[column].fillna(0)
    return df

def fix_snow(df_weather):
        # fix typo values
    df_weather['cumulative_snow_hours'] = df_weather['cumulative_snow_hours'].        apply(lambda x: fix_snow_values(x) if type(x) != float else x)
    return df_weather

def expand_snow_hours(df, column='cumulative_snow_hours'):
    pass

def preprocess_parsed_col(df, column='dew_point'):
    '''
    Redo parsing for dew
    '''
    df[column] = df[column].apply(lambda x: int(x) / 10)
    return df

def remove_missing_wind_speeds(df, column='wind_speed'):
    '''
    Remove wind speeds that are 999.9 (Missing values).
    '''
    df[column] = df[column].apply(lambda x: x if float(x) != 999.9 else 0)
    return df

def categorize_wind_dir(df, column='wind_dir'):
    '''
    Categorize wind dirs into NW, SE, SW, NE, cv (calm and variable)
    '''
    pass

df_weather = preprocess_air_pressure(extra_weather, 'year')

df_weather = fill_wind_dir_nans(df_weather)

df_weather = fix_snow(df_weather)

df_weather = fill_snow_hours_nans(df_weather)

# #Adjusting the parse by 1 decimal place
# df_weather = preprocess_parsed_col(df_weather)
# df_weather = preprocess_parsed_col(df_weather, column='air_pressure')
# df_weather = preprocess_parsed_col(df_weather, column='wind_speed')
# df_weather = preprocess_parsed_col(df_weather, column='air_temp')

pollution_2010_2014.groupby('wnd_dir').count()

df_weather = remove_missing_wind_speeds(df_weather)

df_weather = fill_rain_hours_nans(df_weather)

# df_weather.to_csv('../data/cleaned_weather_2015_2017.csv')

# Querying for data types
# extra_weather['cumulative_snow_hours'].apply(lambda x: type(x)
#                                 )

# querying the dataframe at a particular index:
#extra_weather.iloc[9]




# import necessary modules
import time
from pprint import pprint
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calendar

# load racetrack data that'll be needed for all the results files
rc = pd.read_csv('Racetrack_data.csv', index_col=0)

rc.head()

# define the years we want to combine data for
years = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

# function to parse dates as we read in results csv files
def dateparser(dstr):
    """ Returns a datetime object for any date string in the format
        Month, dd, yyyy """
    d = dict((v,k) for k,v in enumerate(calendar.month_name))
    if type(dstr) != float:
        mon, dd, yyyy = dstr.split(',')
        date_str = '/'.join([str(d[mon]), dd, yyyy])
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    else:
        pass
    return None

# function that'll help determine which string values are actually numbers
def isNumber(x):
    if pd.isnull(x):
        return False
    try:
        float(x)
        return True
    except ValueError:
        pass
    return False

def subset(string, chars):
    if pd.isnull(string):
        return None
    else:
        return string[:chars]

# function that'll convert finishing time into a proper timedelta value
def timeparser(time_series):
    
    new_time = []
    for index, item in enumerate(time_series):
        prev_case = True
        if pd.isnull(item):
            new_time.append(None)
            continue
        
        if 'Lap' in item:
            new_time.append(None)
            prev_case = False
            
        if ('+' in item) & ("'" not in item):
            t = datetime.strptime(item, "+%S.%f")
            t_delta = timedelta(hours=t.hour, minutes=t.minute,seconds=t.second, microseconds=t.microsecond)
            t_delta = t_delta + base_delta
            new_time.append(t_delta)
            prev_case = False
            
        if ('+' in item) & ("'" in item):
            t = datetime.strptime(item, "+%M'%S.%f")
            t_delta = timedelta(hours=t.hour, minutes=t.minute,seconds=t.second, microseconds=t.microsecond)
            t_delta = t_delta + base_delta
            new_time.append(t_delta)
            prev_case = False
            
        if prev_case:
            base_time = datetime.strptime(item, "%M'%S.%f")
            base_delta = timedelta(hours=base_time.hour, minutes=base_time.minute,
                                   seconds=base_time.second, microseconds=base_time.microsecond)
            new_time.append(base_delta)
            
    return new_time

dfs = []
rows_read = []
for yr in reversed(years):
    print(yr, end=', ')
    df = pd.read_csv('/Archive/'+yr+'_data.csv', index_col=0, parse_dates=['Date'], date_parser=dateparser)
    df['Track_Temp'] = df['Track_Temp'].map(lambda x: int(x[:2]) if isNumber(subset(x,2)) else x)
    df['Air_Temp'] = df['Air_Temp'].map(lambda x: int(x[:2]) if isNumber(subset(x,2)) else x)
    df['Humidity'] = df['Humidity'].map(lambda x: float(x[:2])/100 if isNumber(subset(x,2)) else x)
    df['Finish_Time'] = timeparser(df.Time)
    df['GP'] = df.TRK.map(lambda x: x+' - ') + df.Track.map(lambda x: x.split(' - ')[1])
    df = df.merge(rc, on='GP', how='left')
    rows_read.append(len(df))
    dfs.append(df)
print('Complete!')

# save to master CSV
result = pd.concat(dfs, ignore_index=True)
fn = 'MotoGP_2005_2017.csv'
result.to_csv(fn)


# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8

get_ipython().system('ls -lh data/')

# The `wc` shell command shows us how many lines each file has.
# We can see that the `stops.json` file has the most lines (29852).
get_ipython().system('wc -l data/*')

get_ipython().system('head data/Berkeley_PD_-_Calls_for_Service.csv')

calls = pd.read_csv('data/Berkeley_PD_-_Calls_for_Service.csv')
calls

def df_interact(df):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0, col=0):
        return df.iloc[row:row + 5, col:col + 6]
    interact(peek, row=(0, len(df), 5), col=(0, len(df.columns) - 6))
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))

df_interact(calls)

# True if row contains at least one null value
null_rows = calls.isnull().any(axis=1)
calls[null_rows]

# Show the first 7 rows of the table again for reference
calls.head(7)

def combine_event_datetimes(calls):
    combined = pd.to_datetime(
        # Combine date and time strings
        calls['EVENTDT'].str[:10] + ' ' + calls['EVENTTM'],
        infer_datetime_format=True,
    )
    return calls.assign(EVENTDTTM=combined)

# To peek at the result without mutating the calls DF:
calls.pipe(combine_event_datetimes).head(2)

calls['OFFENSE'].unique()

calls['CVLEGEND'].unique()

calls['BLKADDR'][[0, 5001]]

def split_lat_lon(calls):
    return calls.join(
        calls['Block_Location']
        # Get coords from string
        .str.split('\n').str[2]
        # Remove parens from coords
        .str[1:-1]
        # Split latitude and longitude
        .str.split(', ', expand=True)
        .rename(columns={0: 'Latitude', 1: 'Longitude'})
    )

calls.pipe(split_lat_lon).head(2)

# This DF contains the day for each number in CVDOW
day_of_week = pd.read_csv('data/cvdow.csv')
day_of_week

def match_weekday(calls):
    return calls.merge(day_of_week, on='CVDOW')
calls.pipe(match_weekday).head(2)

def drop_unneeded_cols(calls):
    return calls.drop(columns=['CVDOW', 'InDbDate', 'Block_Location', 'City',
                               'State', 'EVENTDT', 'EVENTTM'])

calls_final = (calls.pipe(combine_event_datetimes)
               .pipe(split_lat_lon)
               .pipe(match_weekday)
               .pipe(drop_unneeded_cols))
df_interact(calls_final)

# HIDDEN
# Save data to CSV for other chapters
# calls_final.to_csv('../ch5/data/calls.csv', index=False)


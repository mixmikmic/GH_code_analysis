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

# HIDDEN
def df_interact(df):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0):
        return df[row:row + 5]
    interact(peek, row=(0, len(df), 5))
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))

get_ipython().system('head data/stops.json')

import json

# Note that this could cause our computer to run out of memory if the file
# is large. In this case, we've verified that the file is small enough to
# read in beforehand.
with open('data/stops.json') as f:
    stops_dict = json.load(f)

stops_dict.keys()

from pprint import pformat

def print_dict(dictionary, num_chars=1000):
    print(pformat(dictionary)[:num_chars])

print_dict(stops_dict['meta'])

print_dict(stops_dict['data'], num_chars=300)

# Load the data from JSON and assign column titles
stops = pd.DataFrame(
    stops_dict['data'],
    columns=[c['name'] for c in stops_dict['meta']['view']['columns']])

stops

# Prints column names
stops.columns

columns_to_drop = ['sid', 'id', 'position', 'created_at', 'created_meta',
                   'updated_at', 'updated_meta', 'meta']

# This function takes in a DF and returns a DF so we can use it for .pipe
def drop_unneeded_cols(stops):
    return stops.drop(columns=columns_to_drop)

stops.pipe(drop_unneeded_cols)

# True if row contains at least one null value
null_rows = stops.isnull().any(axis=1)

stops[null_rows]

# True if row contains at least one null value without checking
# the latitude and longitude columns
null_rows = stops.iloc[:, :-2].isnull().any(axis=1)

df_interact(stops[null_rows])

stops['Location'].value_counts()

dispositions = stops['Dispositions'].value_counts()

# Outputs a slider to pan through the unique Dispositions in
# order of how often they appear
interact(lambda row=0: dispositions.iloc[row:row+7],
         row=(0, len(dispositions), 7))

# Strange values...
dispositions.iloc[[0, 20, 30, 266, 1027]]

def clean_dispositions(stops):
    cleaned = (stops['Dispositions']
               .str.strip()
               .str.rstrip(';')
               .str.replace(';', ','))
    return stops.assign(Dispositions=cleaned)

stops_final = (stops
               .pipe(drop_unneeded_cols)
               .pipe(clean_dispositions))
df_interact(stops_final)

# HIDDEN
# Save data to CSV for other chapters
# stops_final.to_csv('../ch5/data/stops.csv', index=False)


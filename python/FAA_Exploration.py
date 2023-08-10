import os
import warnings
import json
import copy
import numpy as np
import pandas as pd
import pandas_profiling
from scipy import stats

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
np.random.seed(42)

# Load data
DIR = './data/original/'

csv_paths = [
    os.path.join(path, f) for path, _, files in os.walk(DIR) for f in files 
    if f.endswith('.csv')]

print('{} files found.'.format(len(csv_paths)))


TYPES = {
    'MONTH': int,
    'DAY_OF_MONTH': int,
    'DAY_OF_WEEK': int,
    'UNIQUE_CARRIER': str,
    'TAIL_NUM': str,
    'ORIGIN': str,
    'ORIGIN_CITY_NAME': str,
    'DEST': str,
    'DEST_CITY_NAME': str,
    'CRS_DEP_TIME': int,
    'DEP_DELAY': float,
    'CRS_ARR_TIME': int,
    'ARR_DELAY': float,
    'CRS_ELAPSED_TIME': float,
    'DISTANCE': float
}

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

dfs = [
    pd.read_csv(
        filepath, sep=',', header=0, dtype=TYPES, 
        infer_datetime_format=True, parse_dates=['FL_DATE'], date_parser=dateparse)
    for filepath in csv_paths
]
print('{} files loaded.'.format(len(dfs)))

df = pd.concat(dfs)

print('Number of columns in training: ', len(df.columns))
print('Number of rows in training: ', len(df))

with pd.option_context('display.max_rows', 3, 'display.max_columns', 300):
    display(df)
    
with pd.option_context('display.max_rows', 100, 'display.max_columns', 300):
    display(df.describe())

# Create some new features

# Use only original departure and arrival hours
df['CRS_DEP_HOUR'] = df['CRS_DEP_TIME'] // 100
df['CRS_ARR_HOUR'] = df['CRS_ARR_TIME'] // 100
print('"CRS_DEP_HOUR" and "CRS_ARR_HOUR" created.')

# Combine destination and origin to create ROUTE
df['ROUTE'] = df[['ORIGIN', 'DEST']].apply(lambda row: '-'.join(row), axis=1)
print('"ROUTE" created.')

# Delete Columns with all Nans
df.dropna(axis=1, how='all', inplace=True)

pandas_profiling.ProfileReport(df, correlation_threshold=1.0)

# Drop missing values
print('Rows before dropping missing values: ', len(df))
df.dropna(axis=0, how='any', inplace=True)
print('Rows after dropping missing values: ', len(df))

# Explore the categorical variables
print('UNIQUE_CARRIER: ', df['UNIQUE_CARRIER'].unique())
print('TAIL_NUM: ', len(df['TAIL_NUM'].unique()))
print('ORIGIN: ', len(df['ORIGIN'].unique()))
print('DEST: ', len(df['DEST'].unique()))
print('ROUTE: ', len(df['ROUTE'].unique()))

print(df['ARR_DELAY'].describe())
sns.distplot(df['ARR_DELAY'])
plt.show()

# There are negative values so we can't use box-cox for normality transformation.
# Use hyperbolic tranformation from "On hyperbolic transformations to normality"
# https://www.sciencedirect.com/science/article/pii/S0167947317301408
# Notice that we seem to get a bi-model Guassian distribution with 0 the 
#  splitting point between the 2 Guassians.
delta = 0.3
eps = 0
delay_transf = np.sinh(delta * np.arcsinh(df['ARR_DELAY']) - eps)
sns.distplot(delay_transf)
plt.show()

# Cube-root also restuls in bimodal distribution
sns.distplot(np.cbrt(df['ARR_DELAY']))
plt.show()

# Clear and save dataset for training
df.to_csv(
    './data/faa_full.csv', header=True, index=False,
    columns=['MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'ORIGIN',
             'DEST', 'ARR_DELAY', 'DISTANCE', 'CRS_ELAPSED_TIME',
             'CRS_DEP_HOUR', 'CRS_ARR_HOUR', 'DEP_DELAY'])


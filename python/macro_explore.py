## usual data analysis imports
import numpy as np
import pandas as pd

macro = pd.read_csv('macro.csv', parse_dates=True, index_col='timestamp')

macro.head()

macro.columns

macro.shape

macro.describe()

macro.dtypes




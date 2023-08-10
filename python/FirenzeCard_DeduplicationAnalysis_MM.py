import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pylab import *

import igraph as ig

import sys
sys.path.append('../../src/')
from utils.database import dbutils

conn = dbutils.connect()
cursor = conn.cursor()

df = pd.read_sql('select * from optourism.firenze_card_logs', con=conn)
df.head()

def frequency(dataframe,columnname):
    out = dataframe[columnname].value_counts().to_frame()
    out.columns = ['frequency']
    out.index.name = columnname
    out.reset_index(inplace=True)
    out = out.sort_values(columnname)
    out['cumulative'] = out['frequency'].cumsum()/out['frequency'].sum()
    out['ccdf'] = 1 - out['cumulative']
    return out

(df['adults_first_use'] + df['adults_reuse'] != df['total_adults']).sum() # Check to make sure the columns add up

(df['total_adults'] > 1).sum() # Check to make sure there is never more than 1 adult per card, acc

fr1 = frequency(df,'minors')
fr1.head() # Only 1 child max per card, which is about 10% of the cases

# Now, do some people visit the same museum more than once?
fr2 = frequency(df.groupby(['museum_name','user_id'])['total_adults'].sum().to_frame(),'total_adults')
fr2.head(20) # Only 19 people visited a place more than once. 

df1 = df.groupby(['museum_name','user_id'])['minors'].sum().to_frame()
fr3 = frequency(df1,'minors')
fr3

df1[df1['minors']>10]

# Check to see what the case of 19 looks like.
df[(df['user_id']==2068648) & (df['museum_name']=='Galleria degli Uffizi')]

df2 = df.groupby(['user_id','museum_name','entry_time']).sum()
df2[(df2['total_adults']>1)|(df2['minors']>1)].head(50)

df[df['user_id']==2017844]


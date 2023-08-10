import os

os.getcwd()

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
#pd.set_option('display.mpl_style', 'default')
#plt.style.use('ggplot')

#pd.set_option('display.line_width', 5000) 
#pd.set_option('display.max_columns', 60) 
#figsize(15, 5)

'''
import csv
with open('C:/Users/skang/data/climate01.csv') as f:
    f_csv = csv.reader(f)   
    
    header = next(f_csv)
    print(header)
    
    for row in f_csv:
       print(row)
'''

'''
df = pd.read_excel( 'xx.xcls')
'''

df01 = pd.read_csv('C:/Users/skang/data/climate01.csv')
df01.head()
pd.to_datetime(df01.Date)
figsize = (15, 10)
plt.plot(pd.to_datetime(df01.Date), df01.Tmax)

#plt.plot(df01.Date.astype('datetime64[ns]'), df01.Tmax)
#plt.show()


    

df01.Date = pd.to_datetime(df01.Date)
df01.iloc[0:5,0:2]
agg01 = df01.groupby('Rad').aggregate(sum)
df01['Tmax'].plot(figsize=(15, 7))

tables = pd.read_html("http://www.basketball-reference.com/leagues/NBA_2016_games.html")
print(tables)
#games = tables[1]
#games.head(10)

#! pip install html5lib

get_ipython().system('jupyter nbconvert data_processing_PANDAS.ipynb --to slides --post serve')

myque = deque(maxlen=2)


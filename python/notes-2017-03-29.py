import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

df_1 = pd.read_csv("gdp_data_raw.csv")

df_1.head()

df_1.info()

df_2 = pd.read_csv("gdp_data.csv")

df_2.info()

df_2['Croatia [HRV]'].head()

df_3 = pd.read_csv("gdp_data.csv",na_values=['..'])

df_3.info()

gdp_data = pd.read_csv("gdp_data.csv",na_values=['..'],index_col=2)

gdp_data.head()

del gdp_data['Series Name']
del gdp_data['Series Code']
del gdp_data['Time Code']

gdp_data.head()

gdp_data.info()

gdp_data.plot(figsize=(15,10))

gdp_data.loc[1995:].mean().sort_values().plot(kind='bar',figsize=(15,5),title='Average GDP per capita since 1995')

gdp_data.corr()

# x=0 means x axis is column 0
# y=1 means y axis is column 1
gdp_data[['Finland [FIN]','Denmark [DNK]']].plot(kind='scatter',x=0,y=1)


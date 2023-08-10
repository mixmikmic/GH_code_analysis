import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

rainfall = pd.read_csv('data/uk_rain_2014.csv', header=0)
rainfall.head()

rainfall.tail()

rainfall.describe()

list(rainfall.columns.values)

rainfall = rainfall.rename(columns = {'\ufeffWater Year':'Water Year'})

def get_base_year(year):
    return int(year[:4])

rainfall['Water Year_num'] = rainfall['Water Year'].apply(get_base_year)

rainfall.head()

rainfall.plot(x='Water Year', y='Rain (mm) Oct-Sep')




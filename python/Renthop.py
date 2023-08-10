get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rent_db = pd.read_json('./train.json')

rent_db.head(5)

rent_db.describe()

rent_db.boxplot(column='price')

rent_db.boxplot(column='price', by='bedrooms')

rent_db.sort_values(by='price',ascending=False, inplace=True)
rent_db.head(20)

rent_db2 = rent_db[rent_db.price < 1000000]
rent_db2.head()

rent_db2.boxplot(column='price')

rent_db3 = rent_db2[rent_db2.price < 7000]
rent_db3.boxplot(column='price')

rent_db3.price.hist(bins=50)

rent_db3.describe()

rent_db3.boxplot(column='price', by='bedrooms')

rent_db.columns

geo_db = rent_db3[['latitude','longitude', 'interest_level']].copy()

geo_db.head()

geo_db.describe()

geo_db.dtypes

def convert_int_lvl(txt_lvl):
    if txt_lvl == 'high':
        int_lvl = 2
    elif txt_lvl == 'medium':
        int_lvl = 1
    elif txt_lvl == 'low':
        int_lvl = 0
    return int_lvl

geo_db['interest_level'] = geo_db['interest_level'].apply(convert_int_lvl)

geo_db.dtypes

geo_db.head()

from pyproj import Proj, transform

inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3857')
geo_db['x'], geo_db['y'] = transform(inProj, outProj, list(geo_db['latitude']), list(geo_db['longitude']))

geo_db.head()

geo_db.isnull().sum()

import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import Greys9
Greys9_r = list(reversed(Greys9))[:-2]

cvs = ds.Canvas(plot_width=600, plot_height=600)
agg = cvs.points(geo_db, 'x', 'y', ds.count('interest_level'))
img = tf.shade(agg, cmap=Greys9_r, how='log')




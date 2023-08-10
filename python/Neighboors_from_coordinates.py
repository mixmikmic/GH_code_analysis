import numpy as np
import pandas as pd

df = pd.read_csv('get_neighborhood.csv')

df.head()

df.drop(['cartodb_id', 'the_geom', 'field_1'],axis=1, inplace=True)

#oops, forgot to drop the neighborhood
df.drop('neighborhood', axis=1, inplace=True)

df.head()

df.shape




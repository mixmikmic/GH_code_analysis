X = [[0], [1], [2], [3]]
y = [0, 1, 9, 9]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 
print(neigh.predict([[0]]))
print(neigh.predict_proba([[0]]))

import datetime as dt
import pandas as pd

df = pd.DataFrame({'date':['2011-04-24 01:30:00.000']})
df

df['date'] = pd.to_datetime(df['date'])
df

(df['date'] - dt.datetime(1970,1,1)).dt.total_seconds()

df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
                    'C': [1, 2, 3]})

pd.get_dummies(df)

import numpy as np

df = pd.DataFrame({'Type':list('ABBC'), 'Set':list('ZZXY')})
df['color'] = np.where(df['Set']=='Z', 'green', 'red')
df

df['texture'] = np.where(df['Type']=='B', 'hard', 'soft')

df

df = pd.DataFrame([['hello', 'hello world'], ['abcd', 'defg']], columns=['a','b'])

df[df['b'].str.contains('hello world')]

df

df

df = pd.DataFrame({'num':list('1233'), 'kc':list('ABBC')})
df['opp'] = np.where(df['kc']=='C', 1, 2)
df

df2 = pd.DataFrame({'num':list('1123'), 'kc':list('BCCD')})
df2['opp'] = np.where(df2['kc']=='C', 2, 3)
df2

df3 = df.merge(df2, on=['num'], how='outer')
df3

df3 = df3.drop_duplicates()

df3

df3['opp'] = df3[['opp_x', 'opp_y']].max(axis=1)

df3 = df3.drop('opp_x', 1)
df3 = df3.drop('opp_y', 1)

df3




import pandas as pd
from pandas import Series, DataFrame

data = Series([4, -1, 3, 2])

data

data[0]

data.values

data.index

data = Series([4, -1, 3, 2], index=['a', 'b', 'c', 'd'])

data

data[0]

data['a']

data[['a', 'b']]

data[data > 0]

data * 2

import numpy as np

np.exp(data)

city_data = {'London': 8.6, 'Paris': 2.2, 'Berlin': 3.6}

data = Series(city_data, index=['Berlin', 'London', 'Madrid', 'Paris', 'Rome'])

data

purchases = [{'Customer': 'Bob', 'Item': 'Oranges', 'Quantity': 2, 'Unit price': 2},
             {'Customer': 'Bob', 'Item': 'Apples', 'Quantity': 3, 'Unit price': 1},
             {'Customer': 'Bob', 'Item': 'Milk', 'Quantity': 1, 'Unit price': 4},
             {'Customer': 'Alice', 'Item': 'Oranges', 'Quantity': 2, 'Unit price': 2},
             {'Customer': 'Alice', 'Quantity': 2, 'Unit price': 3}]
df = DataFrame(purchases)

df

df.loc[0]

df['Item']

df.loc[0, 'Item']

df.loc[0:2, ['Item', 'Quantity']]

is_alice = df['Customer'] == 'Alice'

is_alice

df[is_alice]

df['Total cost'] = df['Unit price'] * df['Quantity']

df

del df['Total cost']

df

df.drop(4)

df

new_df = df.drop(3)
new_df

df.drop(3, inplace=True)
df

df.loc[4, 'Item'] = 'Bananas'

df


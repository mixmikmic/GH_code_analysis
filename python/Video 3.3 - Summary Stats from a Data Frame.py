import pandas as pd

data = pd.read_csv('store_data.csv')

data

data.head()

data['TOTAL'] = data['QUANTITY'] * data['UNIT PRICE']

data.tail()

data.sum()

data[['QUANTITY', 'TOTAL']].sum()

data['UNIT PRICE'].mean()

data['UNIT PRICE'].fillna(0).mean()

data['UNIT PRICE'].median()

data['UNIT PRICE'].fillna(0).sort_values()

data['TOTAL'].max()

data['TOTAL'].argmax()  # index location (int)

data['TOTAL'].idxmax()  # index label

data['ITEM'][8]

data.describe()

data['ITEM'].unique()

data['ITEM'].value_counts()

get_ipython().magic('matplotlib inline')

data['QUANTITY'].hist()




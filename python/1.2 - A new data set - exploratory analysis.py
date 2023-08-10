data_file = '~/data/titanic/train.csv'

import pandas as pd

data = pd.read_csv(data_file)

len(data)

data.head()

data.count()

data.describe()

data[['Age', 'Fare']].describe()

data['Age'].min(), data['Age'].max(), data['Age'].mean()

data['Sex'].value_counts()

data['Sex'].value_counts() / len(data) * 100

data['Survived'].value_counts()

data['Pclass'].value_counts()

data['Pclass'].value_counts().sort_index()

data['Age']. value_counts()

bins = [0, 18, 25, 35, 45, 55, 65, 75, 80]

data['AgeGroup'] = pd.cut(data['Age'], bins)

data['AgeGroup'].value_counts().sort_index()

get_ipython().run_line_magic('matplotlib', 'inline')

data['AgeGroup'].value_counts().sort_index().plot(kind='bar')




import pandas as pd

fname = '~/data/titanic/train.csv'

data = pd.read_csv(fname)

len(data)

data.head()

data.count()

data['Age'].min(), data['Age'].max()

data['Survived'].value_counts()

data['Survived'].value_counts() * 100 / len(data)

data['Sex'].value_counts()

data['Pclass'].value_counts()

get_ipython().magic('matplotlib inline')

alpha_color = 0.5

data['Survived'].value_counts().plot(kind='bar')

data['Sex'].value_counts().plot(kind='bar',
                                color=['b', 'r'],
                                alpha=alpha_color)

data['Pclass'].value_counts().sort_index().plot(kind='bar',
                                                alpha=alpha_color)

data.plot(kind='scatter', x='Survived', y='Age')

data[data['Survived'] == 1]['Age'].value_counts().sort_index().plot(kind='bar')

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]

data['AgeBin'] = pd.cut(data['Age'], bins)

data[data['Survived'] == 1]['AgeBin'].value_counts().sort_index().plot(kind='bar')

data[data['Survived'] == 0]['AgeBin'].value_counts().sort_index().plot(kind='bar')

data['AgeBin'].value_counts().sort_index().plot(kind='bar')

data[data['Pclass'] == 1]['Survived'].value_counts().plot(kind='bar')

data[data['Pclass'] == 3]['Survived'].value_counts().plot(kind='bar')

data[data['Sex'] == 'male']['Survived'].value_counts().plot(kind='bar')

data[data['Sex'] == 'female']['Survived'].value_counts().plot(kind='bar')

data[(data['Sex'] == 'male') & (data['Pclass'] == 1)]['Survived'].value_counts().plot(kind='bar')

data[(data['Sex'] == 'male') & (data['Pclass'] == 3)]['Survived'].value_counts().plot(kind='bar')

data[(data['Sex'] == 'female') & (data['Pclass'] == 1)]['Survived'].value_counts().plot(kind='bar')

data[(data['Sex'] == 'female') & (data['Pclass'] == 3)]['Survived'].value_counts().plot(kind='bar')




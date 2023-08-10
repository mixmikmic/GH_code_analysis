get_ipython().magic('matplotlib inline')
import matplotlib.pylab
import pandas as pd
import numpy as np

students = pd.DataFrame({'phone': ['555-1212', '555-1234', '555-1111', '555-2222'], 'age':[17, 17, 18, 18]}, index = ['Melanie', 'Bob', 'Vidhya', 'Ming'])
students

students.index

df = pd.DataFrame(np.random.randn(6,4), index=['Jenny', 'Frank', 'Wenfei', 'Arun', 'Mary', 'Ivan'], columns=list('ABCD'))
df

s = pd.Series([1,3,5,np.nan,6,8])

s.index = ['a', 'b', 'c', 'd', 'e', 'f']
s

s.isnull()

s.plot()

df.plot()

s[s.index > 'c']

s[s.isnull() == False]

students['age']

students.age

# 'selection by label'
students.loc['Melanie']

students.loc['Melanie', ['age', 'grades']]

# select by position
students.iloc[1, :]

students[students['age'] > 17]

students.age.mean()

students.age.max()

students.age.min()

students['grade'] = [100, 97, 80, 85]

students[students['grade'] == students['grade'].max()]

students.groupby('age').grade.mean()

bins = np.linspace(70, 100, 3)
bins
students.groupby(np.digitize(students.grade, bins)).age.mean()

# First let's see what a lambda function looks like / does
f = lambda x: x + 1

f(4)

students.age.apply(lambda age: age + 1)

students.age.mean()

students.age.count()

students.corr()

students.cummax()


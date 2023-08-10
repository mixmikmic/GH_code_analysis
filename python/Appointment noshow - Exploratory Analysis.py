# Import dependencies
from __future__ import print_function
# Import NumPy
import numpy as np
# Import Pandas
import pandas as pd
# Import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Import
import seaborn as sns

# Read raw dataset
df = pd.read_csv('no-show-hospital-data.csv')

df.head()

df.tail()

df.sample(random_state=1234, n=5)

df.shape

df.PatientId.nunique()

df.groupby('PatientId').PatientId.agg('count').sort_values(ascending=False).head(100)

df.groupby('PatientId').PatientId.agg('count').value_counts()

df.Gender.unique()

df.isnull().sum()

df.dtypes

df.Age.hist(bins=40)
plt.show()

sns.countplot('Gender', data=df)
plt.show()

df.Neighbourhood.unique()

fig, ax = plt.subplots(figsize=(20, 40))
sns.countplot(y='Neighbourhood', data=df)
plt.show()

df.groupby('Neighbourhood').size().sort_values()

sizes = df.groupby('Neighbourhood').size().sort_values()
sizes[sizes < 100]

df.groupby('Neighbourhood').size().hist()
plt.show()

print(df.AppointmentID.nunique(), df.AppointmentID.count())




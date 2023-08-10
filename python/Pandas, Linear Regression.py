import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
get_ipython().magic('matplotlib notebook')

data = pd.read_csv('datasets/Ecdat/BudgetFood.csv')
data = data.drop('Unnamed: 0', axis=1)
data.head()

# View the distributions of your dataset as histograms
data.hist()

group = data.groupby(['sex', 'town'])
group = group.mean()
group = group.reset_index()
group

ax = group[group.sex == 'man'].plot(x='age', y='totexp', label='man')
group[group.sex == 'woman'].plot(x='age', y='totexp', label='woman', ax=ax)
group.plot(kind='scatter', x='age', y='totexp')

x = np.matrix(group.age).T
y = np.matrix(group.totexp).T
test_x = np.matrix(np.linspace(45, 70, 10)).T

# Create a linear regression model
regr = LinearRegression()
model = regr.fit(x, y)
results = model.predict(test_x)
print(results)

plt.figure()
plt.scatter(x, y)
plt.plot(test_x, results)
plt.show()


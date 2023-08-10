from sml import execute

query = 'READ "../data/iris.csv"'

execute(query, verbose=True)

import pandas as pd

names = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)', 'species']
data = pd.read_csv('../data/iris.csv', names=names)
data.head()




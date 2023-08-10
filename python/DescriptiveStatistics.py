import pandas as pd

# View first 20 rows
filename = "../data/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# df stands for "Data Frame"
df = pd.read_csv(filename, names=names)
pd.set_option('precision', 3)
df.describe()

df.groupby('class').size()

df.corr(method='pearson')

df.skew()

df = pd.read_csv('../data/winequality-red.csv', sep=';')
df.describe()

df.groupby('quality').size()

df.corr(method='pearson')

df.skew()


import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().magic('matplotlib inline')

get_ipython().system('pwd')

df = pd.read_csv("claims-time.csv", index_col=0, parse_dates=True)

print(df.head())
#this time we parse from the begining

df.plot()
plt.title("Number of Claims")
plt.ylabel("Claims")
plt.grid(True)
plt.show

df_sort = df.sort_values(['Claims'], ascending=False)
df_sort.count()

print(df_sort.head())
#the days with more claims in history

#let's see all days in negatives we have
df[df['Claims'] < 0].count()

df.describe()

#The rule we gonna use is excluding all the data with days more than 2 estandart deviatin_
two_dev = 2*df.std().astype(int)
two_dev

df1 = df[df['Claims'] < 566]
df1.count()

#We have 4595 from 5613 
df1.count()/df.count()*100

mpl.style.use('ggplot')
df1.plot()
plt.title("Number of Claims")
plt.ylabel("Claims")
plt.grid(True)
plt.show

df1.describe()

#pimped
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
print(df1.describe().unstack()) #unstack makes a table if we have many features

#Some more optional features

#In case we have classes
class_counts = data.groupby('class').size()
print(class_counts)

#correlations
correlations = data.corr(method='pearson')
print(correlations)

skew = data.skew()
print(skew)

#print(df.mean(),df1.mean())
print((df1.mean()/df.mean())-1)

df.dtypes
#it is such a shame to notice here that the variable In is an object, we need datetime instead

df1['Claims']

dfy = df1.groupby(df1['Claims'].map(lambda x: x.year)).mean()
dfy.head()

dfm.plot()
plt.title("Average Hospitalization Days")
plt.ylabel("Avg Day")
plt.grid(True)
plt.show

dfy = df3.groupby(df3['In'].map(lambda x: x.year)).mean()
dfy.head()

dfy.plot()
plt.title("Average Hospitalization Days")
plt.ylabel("Avg Day")
plt.grid(True)
plt.show

#this is just an example
mpl.style.available
[u'dark_background', u'grayscale', u'ggplot']
mpl.style.use('ggplot')
plt.hist(np.random.randn(100000))




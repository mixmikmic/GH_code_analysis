import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().magic('matplotlib inline')

get_ipython().system('pwd')

df = pd.read_csv("hospit-days.csv", parse_dates=True)

print(df[0:10])

DateIn = df["In"]
AvgDays = df["Days"]

#This is a kind of plot that depends on a X axis and Y axis
plt.plot_date(DateIn,AvgDays, ls='--')
plt.show

#Parsing the vector
raw_data = {'In': DateIn, 'Days': AvgDays}
df0 = pd.DataFrame(raw_data, columns = ['In', 'Days'])
#df0.head()

df0.plot()
plt.title("Average Hospitalization Days")
plt.ylabel("Avg Day")
plt.grid(True)
plt.show

#how the top10 looks like
df0_sort = df0.sort_values(['Days'], ascending=False)
print(df0_sort.head(10))

#let's see all days in negatives we have
df0[df0['Days'] < 0].count()

#New dataframes without negatives
df1 = df0[df0['Days'] > 0]

df1.plot()
plt.title("Average Hospitalization Days")
plt.ylabel("Avg Day")
plt.grid(True)
plt.show

#Not so many of those
df0[df0['Days'] > 365].count()

#Set newdataframe fltered
df2 = df0[(df0['Days'] < 366) & (df0['Days'] > 0)]

print(df2.shape)

df2.plot()
plt.title("Average Hospitalization Days")
plt.ylabel("Avg Day")
plt.grid(True)
plt.show

#summary
print(df2.describe())

#The rule we gonna use is excluding all the data with days more than 2 estandart deviatin_
two_dev = 2*df2.std().astype(int)
two_dev

df3 = df2[df2['Days'] < 22]
df3.count()

#not so clear but it means averything around 2 std dev
mpl.style.use('ggplot')
df3.plot()
plt.title("Average Hospitalization Days")
plt.ylabel("Avg Day")
plt.grid(True)
plt.show

df3.count()/df0.count()
#this is the percentage of the data contained within 2 standart deviation

df3.describe()

df3.dtypes
#it is such a shame to notice here that the variable In is an object, we need datetime instead

df3['In'] = pd.to_datetime(df3['In'])
#the error is telling me .loc[row_indexer,col_indexer] = value instead is better

df3.dtypes
#nos we have datetime as we wanted

#pd.Series(np.random.randn(150).cumsum(), index=pd.date_range('2000-1-1', periods=150, freq='B'))
#just an example of series generated

#x, y  = df3['In'], df3['Days']
#fig, ax = plt.subplots()
#ax.plot_date(x, y, linestyle='--')
#plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#This is just an example of annotation
#ax.annotate('Test', (mdates.date2num(x[1]), y[1]), xytext=(15, 15), 
#            textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
#fig.autofmt_xdate()
#plt.show()

months = df3['In'].dt.to_period("M")
dfm = df3.groupby(months).mean()
print(dfm.head())

dfm.plot()
plt.title("Average Hospitalization Days")
plt.ylabel("Avg Day")
plt.grid(True)
plt.show

dfy = df3.groupby(df3['In'].map(lambda x: x.year)).mean()
dfy

dfy.plot()
plt.title("Average Hospitalization Days")
plt.ylabel("Avg Day")
plt.grid(True)
plt.show




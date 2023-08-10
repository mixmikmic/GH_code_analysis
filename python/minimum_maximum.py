import pandas as pd
import numpy as np
from statsmodels.graphics import tsaplots
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

#take the Date and Time strings and convert into one datetime column
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M')

fn = 'Full_Electric_Interval_042016.csv'
chunksize = 10000
dfc = pd.read_csv(fn, chunksize=chunksize,header=3, parse_dates={'datetime': ['Date', 'Start Time']},                   date_parser=dateparse,index_col=0)

df = pd.concat([chunk for chunk in dfc])

df.Meter.unique()

df1 = df.copy()

df1.Meter.unique()

df2 = pd.concat([df_final, usage])



df1.Meter = df1.Meter.map(lambda x: " ".join(x.split()[:-1]))

crerar=df1[df1.Meter == 'A06 Crerar Library'] 
usage = crerar.resample("H", label = 'right', closed = 'right').apply({'Meter':'max','Usage': sum, 'Temperature':'mean'})

df_final = pd.DataFrame()

df.Meter.unique()

hinds = []
for meter in df.Meter.unique():
    if 'Hinds' in meter:
        hinds.append(meter)

for meter in df.Meter.unique():
    df.Usage[df.Meter == meter].resample("H").sum()
    df.Temperature[df.Meter == meter].resample("H").mean()
    

crerar = df[df.Meter.isin(hinds)]

crerar

crerardaily = crerar.resample('H', how = 'mean')

crerarmax = crerar.resample('H', how = 'max')

crerarmin = crerar.resample('H', how = 'min')

minmax = {'Max Usage': crerarmax.Usage, 'Avg Usage' : crerardaily.Usage, 'Temp' : crerardaily.Temperature}

crerarminmax = pd.DataFrame(minmax, index = crerardaily.index)

#Fluctuations of minimum and maximum daily usage compared to average daily temperature.
ax=crerarminmax[-24:].plot(y=['Max Usage', 'Avg Usage', 'Temp'],color=['red', 'blue', 'black'])
fig = ax.get_figure()
ax.set_title('UChicago - Crerar Library - Daily Max/Min Usage')
ax.set_ylabel('Electricity Usage [kWh]')
fig.set_figwidth(20)
fig.set_figheight(10)
# fig.savefig('plots/crerar_daily.png')
# plt.close(fig)

fig = tsaplots.plot_acf(crerardaily.Usage[-60:], ax = None)
fig.set_figwidth(20)
fig.set_figheight(5)

fig = tsaplots.plot_pacf(crerardaily.Usage[-60:], ax = None)
fig.set_figwidth(20)
fig.set_figheight(5)

dates = ['2014-04-10', '2014-06-04', '2014-06-14',
'2014-06-23', '2014-08-31', '2014-09-28', '2014-12-03',
'2014-12-13', '2015-01-04', '2015-03-11', '2015-03-21',
'2015-03-29', '2015-06-03', '2015-06-13', '2015-06-21',
'2015-08-29', '2015-09-27', '2015-12-03', '2015-12-12',
'2016-01-03', '2016-03-09', '2016-03-19', '2016-03-27',
'2016-05-10']

labels = ['Spring 14', 'Reading/Exam Spring 14', 'Break Spring 14',
'Summer 14', 'Break Summer 14', 'Fall 14', 'Reading/Exam Fall 14',
'Break Fall 14', 'Winter 15', 'Reading/Exam Winter 15', 'Break Winter 15',
'Spring 15', 'Reading/Exam Spring 15', 'Break Spring 15', 'Summer 15',
'Break Summer 15', 'Fall 15', 'Reading/Exam Fall 15', 'Break Fall 15',
'Winter 16', 'Reading/Exam Winter 16', 'Break Winter 16', 'Spring 16']

print(len(dates))
print(len(labels))

i = 0
uniquedays = []
terms = []
for date in df1.index.unique():
    if (str(date) > dates[i]) and (str(date) <= dates[i + 1]):
        terms.append(labels[i])
        uniquedays.append(date)
    else:
        i += 1
        print(date)
        print(i)
        terms.append(labels[i])
        uniquedays.append(date)

termlabels = dict(zip(uniquedays, terms))

df['Term'] = df['Date'].map(termlabels)


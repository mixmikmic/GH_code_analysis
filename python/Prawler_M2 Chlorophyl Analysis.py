get_ipython().magic('matplotlib inline')

import datetime
import pandas as pd

f='data/Intchlor.csv' #previously processed Prawler data combined with EcoFluorometer data

data = pd.read_csv(f,index_col=0,parse_dates=True)

data.info()
data.head()

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker

plt.style.use('bmh')

def axis_formater(ax):
    ax.xaxis.set_major_locator(dates.DayLocator(bymonthday=15))
    ax.xaxis.set_minor_locator(dates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.legend()
    ax.set_xlim([datetime.datetime(2016,5,1),datetime.datetime(2016,10,1)])
    return (ax)

plt.figure(figsize=(16,8))
plt.subplot(3,1,1)
ax=plt.gca()
plt.plot(data.index,data['IntegratedChlor_5t045'])
ax = axis_formater(ax)

plt.subplot(3,1,2)
ax=plt.gca()
plt.plot(data.index,data['11m Prawler'])
plt.plot(data.index,data['11m Eco'])
ax = axis_formater(ax)

plt.subplot(3,1,3)
ax=plt.gca()
plt.plot(data.index,data['24m Prawler'])
plt.plot(data.index,data['24m Eco'])

ax = axis_formater(ax)

ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
ax.xaxis.set_tick_params(which='major', pad=15)


def axis_formater(ax):
    ax.xaxis.set_major_locator(dates.DayLocator(bymonthday=15))
    ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.legend()
    ax.set_xlim([datetime.datetime(2016,5,1),datetime.datetime(2016,6,1)])
    return (ax)

plt.figure(figsize=(16,8))
ax = plt.subplot(3,1,1)
plt.plot(data.index,data['IntegratedChlor_5t045'])
ax = axis_formater(ax)
ax.set_xlim([datetime.datetime(2016,5,1),datetime.datetime(2016,6,1)])
plt.subplot(3,1,2)
ax=plt.gca()
plt.plot(data.index,data['11m Prawler'])
plt.plot(data.index,data['11m Eco'])
ax = axis_formater(ax)
ax.set_xlim([datetime.datetime(2016,5,1),datetime.datetime(2016,6,1)])
plt.subplot(3,1,3)
ax=plt.gca()
plt.plot(data.index,data['24m Prawler'])
plt.plot(data.index,data['24m Eco'])

ax = axis_formater(ax)
ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
ax.xaxis.set_tick_params(which='major', pad=15)


plt.figure(figsize=(16,8))
ax = plt.subplot(3,1,1)
plt.plot(data.index,data['IntegratedChlor_5t045'])
ax = axis_formater(ax)
ax.set_xlim([datetime.datetime(2016,5,1),datetime.datetime(2016,6,1)])
plt.subplot(3,1,2)
ax=plt.gca()
plt.plot(data.index,data['0m Eco'])
plt.plot(data.index,data['11m Eco'])
plt.plot(data.index,data['24m Eco'])
plt.plot(data.index,data['52m Eco'])
ax = axis_formater(ax)
ax.set_xlim([datetime.datetime(2016,5,1),datetime.datetime(2016,6,1)])
plt.subplot(3,1,3)
ax=plt.gca()
plt.plot(data.index,data['0m Eco'])
plt.plot(data.index,data['11m Prawler'])
plt.plot(data.index,data['24m Prawler'])
plt.plot(data.index,data['52m Eco'])

ax = axis_formater(ax)
ax.set_xlim([datetime.datetime(2016,5,1),datetime.datetime(2016,6,1)])
ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
ax.xaxis.set_tick_params(which='major', pad=15)




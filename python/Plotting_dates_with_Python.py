import datetime

today = datetime.date.today()
print today
print 'ctime:', today.ctime()
print 'tuple:', today.timetuple()
print 'ordinal:', today.toordinal()
print 'Year:', today.year
print 'Mon :', today.month
print 'Day :', today.day

testdate = datetime.date(2016,6,11)
print(testdate.year)
print(testdate.month)
print(testdate.day)

dir(testdate)

# find out what day it was on a particular date
testdate = datetime.date(2016,6,11)
print(testdate.weekday())

# compare which date was earlier
testdate = datetime.date(2016,6,11)
testdate1 = datetime.date(2016,6,18)
testdate1< testdate

# get today's date
today = datetime.date.today()
type(today)
# number of days from beginning of Gregorian calendar
print("proleptic Gregorian ordinal",today.toordinal())

# form date from integer date
another=datetime.date.fromordinal(736126)
print(another)

# find date 1000 days from now
today = datetime.date.today()
print( datetime.date.fromordinal(today.toordinal()+1000))

# here are the limits of years that can be represented
print(datetime.MINYEAR)
print(datetime.MAXYEAR)

# plotting is always done by converting date to ordinal
# the trick is to hide the ordinal number and present human readable date as tick
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import drange
from numpy import arange,zeros
get_ipython().magic('matplotlib inline')

date1 = datetime.datetime(1916, 1, 1)
date2 = datetime.datetime(1916, 12, 31)
delta = datetime.timedelta(days=1)
dates = drange(date1, date2, delta)
data = zeros(len(dates))


daten = datetime.datetime(1916, 9, 1)
data[daten.toordinal()-date1.toordinal()]=7

fig, ax = plt.subplots()
ax.plot_date(dates, data)

#ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
fig.autofmt_xdate()

plt.show()




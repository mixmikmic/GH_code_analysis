import matplotlib
import matplotlib.pyplot as plt
from climata.usgs import DailyValueIO
import pandas as pd
import numpy as np

plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (14.0, 8.0)

# set parameters
nyears = 10
ndays = 365 * nyears
station_id = "06730200"
param_id = "00060"

datelist = pd.date_range(end=pd.datetime.today(), periods=ndays).tolist()
data = DailyValueIO(
    start_date=datelist[0],
    end_date=datelist[-1],
    station=station_id,
    parameter=param_id,
)

# create lists of date-flow values
for series in data:
    flow = [r[0] for r in series.data]
    dates = [r[1] for r in series.data]

plt.plot(dates, flow)
plt.xlabel('Date')
plt.ylabel(series.variable_name)
plt.title(series.site_name)
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.show()

# set parameters
nyears = 1
ndays = 365 * nyears
county = "08013"
datelist = pd.date_range(end=pd.datetime.today(), periods=ndays).tolist()

data = DailyValueIO(
    start_date=datelist[0],
    end_date=datelist[-1],
    county=county,
)

date = []
value = []

for series in data:
    for row in series.data:
        date.append(row[1])
        value.append(row[0])

site_names = [[series.site_name] * len(series.data) for series in data]

# unroll the list of lists
flat_site_names = [item for sublist in site_names for item in sublist]

# bundle the data into a data frame
df = pd.DataFrame({'site': flat_site_names, 
                   'date': date, 
                   'value': value})

# remove missing values
df = df[df['value'] != -999999.0]

# visualize flow time series, coloring by site
groups = df.groupby('site')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.date, np.log(group.value), marker='o', linestyle='-', ms=2, label=name)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Date')
plt.ylabel('log(' + series.variable_name + ')')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.show()


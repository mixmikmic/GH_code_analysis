get_ipython().magic('matplotlib inline')

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json
import datetime
import scipy.stats

matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12.0, 8.0]

with open('logs/2016-07-01.json') as f:
    sample_data = json.loads(f.read())
list(sample_data.keys())

list(sample_data['summary'].keys())

list(sample_data['sleep'][0].keys())

dates = pd.date_range('2016-03-29', '2016-06-10')
time_in_bed = []

for date in dates:
    fname = 'logs/' + date.strftime('%Y-%m-%d') + '.json'
    with open(fname) as f:
        date_data = json.loads(f.read())
        
        time_in_bed.append(date_data['summary']['totalTimeInBed'] / 60.0)
        
df = pd.DataFrame(time_in_bed, index = dates)
df.columns = ['bed']

df.plot()
plt.ylabel('Hours in Bed');

df.describe().transpose()

df.plot.hist(bins = 8, range = (3, 11))

plt.xlim(3, 11)
plt.xticks(range(3, 11))
plt.xlabel('Hours in Bed')
plt.ylabel('Count');

df['day_of_week'] = df.index.weekday
df['day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
df.head()

df.boxplot(column = 'bed', by = 'day_type', positions = [2, 1], 
           vert = False, widths = 0.5)
plt.xlabel('Hours in Bed')
plt.suptitle('')
plt.title('');

# Group dataframe by weekday vs. weekend
df_weekdays = df[df.day_of_week < 5]
df_weekend = df[df.day_of_week >= 5]

scipy.stats.ttest_ind(df_weekdays['bed'], df_weekend['bed'])

# Add a label for day name, to make the boxplot more readable
days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 
        5: 'Saturday', 6: 'Sunday'}
df['day_name'] = df['day_of_week'].apply(lambda x: days[x])

df.boxplot(column = 'bed', by = 'day_name', positions = [5, 1, 6, 7, 4, 2, 3])

# Configure title and axes
plt.suptitle('')
plt.title('')
plt.ylabel('Hours in Bed')
plt.xlabel('');

bedtimes = []

# Read data into list
for date in dates:
    fname = 'logs/' + date.strftime('%Y-%m-%d') + '.json'
    with open(fname) as f:
        date_data = json.loads(f.read())
        
        # Note that sleep_event['startTime'][11:16] gets the hh:mm characters
        # from the start of a sleep event; it is then converted to a datetime
        for sleep_event in date_data['sleep']:
            bedtimes.append((pd.to_datetime(sleep_event['startTime'][11:16]), 
                             sleep_event['timeInBed'] / 60.0,
                             sleep_event['isMainSleep']))
            
# Convert to dataframe, and make 'bedtime' a float (e.g., 5:30 -> 5.5)
df = pd.DataFrame(bedtimes, columns = ['bedtime', 'duration', 'main'])
df['bedtime'] = df['bedtime'].dt.hour + df['bedtime'].dt.minute / 60.0

# Make first plot: scatterplot of bedtime vs. duration, colored by main sleep
ax = df[df.main == True].plot.scatter(x = 'bedtime', y = 'duration', 
                                      color = 'Red', s = 100, 
                                      label = 'Main Sleep')

df[df.main == False].plot.scatter(x = 'bedtime', y = 'duration', 
                                  color = 'SlateBlue', ax = ax, s = 100, 
                                  label = 'Secondary')

# List of times to use for labels
times = [str(2 * h) + ':00' for h in range(12)]

# Configure legend, x-axis, labels
plt.legend(scatterpoints = 1, markerfirst = False)
plt.xticks(range(0, 24, 2), times)
plt.xlim(0, 24)
plt.xlabel('Bedtime')
plt.ylabel('Hours in Bed');

# Overlay a histogram of bedtimes on the same plot, using a secondary y-axis
ax2 = ax.twinx()
df['bedtime'].map(lambda x: int(x)).plot.hist(bins = range(24), 
                                              color = 'MediumSeaGreen', 
                                              alpha = 0.3, grid = False)

# Configure secondary y-axis
plt.yticks(range(0, 28, 4))
plt.ylim(0, 28);


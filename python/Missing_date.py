import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from datetime import datetime
# %matplotlib inline

# Load the dataset. This dataset provides the information about busline, date, and how many records in a certain day
# for a specific busline.
gap_count = pd.read_csv('count.csv')
gap_count['Date'] = pd.Series(gap_count['Date'])
gap_count.head()

# Figure out how many different buslines are there in the whole dataset.
linelist = list(set(gap_count['Busline']))
print linelist
len(linelist)

# Find out the how many different days' records are there included in the dataset.
timelist = list(set(gap_count['Date']))
print timelist
len(timelist)

days_of_month = [31,28,31,30,31,30,31,31,30,31,30,31]
for i in range(12):
    for j in range(days_of_month[i]):
        tmptimelist = str(i+1) + '/' + str(j+1) + '/15'
        if tmptimelist not in timelist:
            print tmptimelist


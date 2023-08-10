from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
from datetime import datetime
get_ipython().magic('matplotlib inline')

rows = open('data/reversals_raw_mwrd.txt', 'r').read().split('\n');
rows[:15]

# We are expecting 29 events, each represented by 5 rows
num_events = len(rows)/5
print("Number of events found: %s" % str(num_events))

# MWRD gives us the date in many formats.  Normalize it.
def normalize_date(date_str):
    date_split = date_str.split('/')
    end_month = None    
    end_day = None
    day = date_split[1]
    if '-' in day:
        start_day = day.split('-')[0]
        if len(date_split) == 3:
            end_day = day.split('-')[1]
        else:
            end_month = day.split('-')[1]
            end_day = date_split[2]
    else:
        start_day = day
    year = int(date_split[-1])
    if year < 1985:
        if year > 20:
            year = '19%s' % year
        elif year > 9:
            year = '20%s' % year
        else:
            year = '200%s' % year
    start_date = datetime(int(year), int(date_split[0]), int(start_day))
    end_date = None
    if end_day is not None:
        if end_month is not None:
            end_date = datetime(int(year), int(end_month), int(end_day))
        else:
            end_date = datetime(int(year), int(date_split[0]), int(end_day))
    return{'start_date': start_date, 'end_date': end_date,
            'year': int(year)}
date_str = '4/18-5/1/13'
print(date_str)
normalize_date(date_str)

# Build a dict with the attributes in the raw file, which come in 5 row blocks
events = []
for i in range(int(num_events)):
    date_str = rows[(i*5)]
    normalized_date = normalize_date(date_str)
    events.append({
                'date_raw': date_str,
                'obrien': float(rows[(i*5) + 1].strip()),
                'crcw': float(rows[(i*5) + 2].strip()),
                'wilmette': float(rows[(i*5) + 3].strip()),
                'total': float(rows[(i*5) + 4].strip()),
                'start_date': normalized_date['start_date'],
                'end_date': normalized_date['end_date'],
                'year': normalized_date['year'],
            
    })
events[:3]

# Create dataframe with CSOs from river reversal here.  From this point forward, CSO refers to CSOs into the lake only.
csos = pd.DataFrame(events)
csos.head()

# Let's see the number of CSOs by year
csos_by_year = dict(csos['year'].value_counts())
print(csos_by_year)

# Add the years that are missing with 0 CSOs
for year in range(1985,2016):
    if year not in csos_by_year.keys():
        csos_by_year[year] = 0
csos_by_year

# Plot the number of CSOs per year
plt.bar(csos_by_year.keys(), csos_by_year.values(), 1/1.5, color="blue")

# We have dumped 46 billion gallons of sewage into Lake Michigan since 1985
csos[['total']].sum()

# Compare the total sewage dumped per outfall point
totals_by_lock = pd.DataFrame(csos[['crcw', 'obrien', 'wilmette']].sum(axis=0), columns=['sewage_dumped'])
totals_by_lock

# Plot the percentage of total sewage dumped, per outfall point
plt.pie(totals_by_lock.sewage_dumped.values, labels=totals_by_lock.index.values,
                autopct='%1.1f%%', shadow=True, startangle=90)

# Find the CSOs per year per outfall point
csos_by_year_loc = csos.groupby(['year']).sum()[['crcw', 'obrien', 'wilmette']]
csos_by_year_loc.head()

# Plot outfall volume per outfall point per year
csos_by_year_loc.plot(kind='bar')

# Create a CSV file for reversals into Lake Michigan
csos.to_csv('data/lake_michigan_reversals.csv', index=False)


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set larger font size
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

import os

files = os.listdir("/data/measurements")

# Print the files on one line each.
print('\n'.join(files))

import pandas as pd

df = pd.read_csv('/data/measurements/C47C8D65CB0F.csv',
                 names=['time', 'moisture', 'temperature', 'conductivity', 'light'])
print(df.to_string())

from datetime import datetime

x = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df['time']]
y = df['moisture']

plt.plot(x, y, label='C47C8D65CB0F');

fig, ax = plt.subplots(figsize=(8, 6))

# 'b-' stands for blue line
ax.plot(x, y, 'b-', label='C47C8D65CB0F')

# Show legend
ax.legend()

# Rotate the time axis labels and show them as month and day
plt.xticks(rotation=45)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Show axis labels
plt.xlabel('time')
plt.ylabel('moisture');

# Select data
df_sel = df[df['time'] > '2018-02-02 08:30:00']

# The remainig lines are the same as above
x = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df_sel['time']]
y = df_sel['moisture']

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, 'b-', label='C47C8D65CB0F')
ax.legend()
plt.xticks(rotation=45)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xlabel('time')
plt.ylabel('moisture');




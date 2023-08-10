import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

fpath = './data/cardioActivities.csv'
df = pd.read_csv(fpath)

print('Number of activities = {:d}'.format(len(df)))
df.head()

df['Duration'] = df['Distance (mi)']/df['Average Speed (mph)']*60.0  # Duration in minutes
df['Average Pace'] = 60.0/df['Average Speed (mph)']  # Pace in minutes/mile

df = df[df['Type'] == 'Running']  # Keep only running activies
df = df.dropna(subset=['GPX File'])  # Drop records without a GPX file

print('Number of GPS-tracked running activities = {:d}'.format(len(df)))
df.head()



df['Distance (mi)'].hist(bins=20)

plt.scatter(df['Distance (mi)'], df['Average Pace'])
plt.xlim(0, 30)
plt.xlabel('Distance (mi)')
plt.ylim(7, 12.5)
plt.ylabel('Pace (min/mi)')
plt.show()

plt.scatter(df['Distance (mi)'], df['Average Pace'])
plt.xscale('log')
plt.xlim(0, 20)
plt.xlabel('Distance (mi)')
plt.yscale('log')
plt.ylim(7, 12.5)
plt.ylabel('Pace (min/mi)')
plt.show()

df.max()




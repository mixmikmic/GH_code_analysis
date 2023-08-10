import pandas as pd
import os

dirPath = os.path.realpath('.')
fileName = 'assets/coolingExample.xlsx'
filePath = os.path.join(dirPath, fileName)

df = pd.read_excel(filePath,header=0)
df.head()

df[df.columns[0]]

try:
    df[1]
except KeyError:
    print("KeyError: 1 - not a valid key")

cols = df.columns
for col in cols:
    print(df[col])

import matplotlib.pyplot as plt

plt.figure(1)
ax = df.plot()
plt.show()

plt.figure(2)
ax = df.plot(cols[0],cols[1])
plt.show()

plt.figure(3)
ax = df.plot(cols[0],cols[1])
ax.set_title('This is a Title')
ax.set_ylabel('Temperature (deg F)')
ax.grid()
plt.show()

df[cols[0]][0]

from datetime import datetime, date

startTime = df[cols[0]][0]
timeArray = []
for i in range(0,len(df[cols[0]])):
    timeArray.append((datetime.combine(date.today(), df[cols[0]][i]) - datetime.combine(date.today(), startTime)).total_seconds())

plt.figure(4)
plt.plot(timeArray, df[cols[1]], 'b')
plt.title('This is a graph with a better time axis')
plt.ylabel('Temperature (deg F)')
plt.xlabel('Time (s)')
plt.grid()
plt.show()


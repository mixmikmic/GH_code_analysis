import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('OECDBLI2017cleanedcsv.csv')
data.head()

plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plt.hist(data['Household net adjusted disposable income in usd'], bins=20, color='red', alpha=0.5)
plt.hist(data['Household net financial wealth in usd'], bins=70, color='blue', alpha=0.5)
plt.title('1. Household net adjusted disposable income vs\nHousehold net financial wealth in usd')
plt.xlabel('USD')
plt.ylabel('Count')
plt.xticks(rotation=90)

plt.subplot(1,2,2)
plt.boxplot(data['Household net financial wealth in usd'])
plt.title('2. Household net financial wealth')
plt.ylabel('USD')

plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))

plt.scatter(data['Water quality as pct'], data['Air pollution in ugm3'], marker='x', color='orange')
plt.title('1. Water quality vs Air pollution')
plt.xlabel('Water quality in Percentage (%)')
plt.ylabel('Air pollution in Micrograms \nper Cubic Meter of Air (µg/m3)')

plt.show()

country_list = list(data['Country'])

plt.figure(figsize=(10,5))

y = data['Employment rate as pct']
y2 = data['Labour market insecurity as pct']
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38])

plt.xticks(x, country_list)
p1 = plt.plot(x, y, label='Employment rate', color='green')
p2 = plt.plot(x, y2, label='Labour market insecurity', color='deepskyblue')
plt.xticks(rotation=90)
plt.title('1. Correlation between employment rate and labour market insecurity')
plt.xlabel('Country')
plt.ylabel('Percentage (%)')
plt.legend()

plt.show()

plt.figure(figsize=(5,5))
plt.scatter(data['Life satisfaction as avg score'], data['Self-reported health as pct'], marker='x', color='mediumvioletred')
plt.title('1. Life satisfaction vs Self-reported health')
plt.xlabel('Life satisfaction in Average score')
plt.ylabel('Self-reported health in Percentage (%)')

plt.show()


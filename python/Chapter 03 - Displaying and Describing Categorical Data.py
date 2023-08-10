# %load ./snippets/data-imports
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set_style('whitegrid')

# use `value_counts` on a series to get a basic frequency table
# (sort on the index to place the categroies in sorted alpha/numeric order)
d1 = ['c1', 'c1', 'c1', 'c2', 'c2', 'c2', 'c2', 'c2', 'c2', 'c3', 'c3', 'c3',  'c3', 'c3', 'c3',  'c3', 'c3', 'c3']
s1 = Series(d1)
s1.value_counts().sort_index(0)

# add `(normalize=True)` to generate relative frequencies (pcts)
s1.value_counts(normalize=True).sort_index(0)

# draw a basic bar chart of counts
freq_table = s1.value_counts().sort_index(0)
objects = freq_table.index.values
y_pos = np.arange(len(objects))
performance = freq_table.values
 
plt.bar(y_pos, performance, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('# of Instances')
plt.title('Category Counts')
 
plt.show()

# draw a basic bar chart of frequencies
freq_table = s1.value_counts(normalize=True).sort_index(0)
objects = freq_table.index.values
y_pos = np.arange(len(objects))
performance = freq_table.values * 100
 
plt.bar(y_pos, performance, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('% of Instances')
plt.title('Category Relative Frequencies')
 
plt.show()

# pandas offers similar plotting, but with less direct control over some of the params
# note: you can grab the returned plot object and set various properties on it
freq_table.plot.bar(title='test');

labels = freq_table.index.values
values = freq_table.values
patches, texts = plt.pie(values)
plt.legend(patches, labels)
plt.axis('equal')
plt.show()

# load the contingencyTable function
get_ipython().magic('run ./scripts/contingencytable.py')

data = [
  { 'y': 'y1', 'x': 'x1' },
  { 'y': 'y2', 'x': 'x2' },
  { 'y': 'y2', 'x': 'x3' },
  { 'y': 'y3', 'x': 'x1' },
  { 'y': 'y3', 'x': 'x2' },
  { 'y': 'y3', 'x': 'x3' },
  { 'y': 'y1', 'x': 'x1' },
  { 'y': 'y2', 'x': 'x2' },
  { 'y': 'y2', 'x': 'x3' },
  { 'y': 'y3', 'x': 'x2' },
  { 'y': 'y3', 'x': 'x2' },
  { 'y': 'y3', 'x': 'x3' },
  { 'y': 'y1', 'x': 'x1' },
  { 'y': 'y2', 'x': 'x2' },
  { 'y': 'y2', 'x': 'x3' },
  { 'y': 'y3', 'x': 'x1' },
  { 'y': 'y1', 'x': 'x3' }            
    ]
df = DataFrame(data=data)
ct = contingencyTable(df, 'y', 'x')
ct

# as an example, let's look at just the y2 values above
ctTemp = ct.drop('All', axis=1)
ctTemp.head()

ct.loc['y2', 'pctOfColumn']

y1 = ctTemp.loc['y1', 'pctOfColumn']
y2 = ctTemp.loc['y2', 'pctOfColumn']
y3 = ctTemp.loc['y3', 'pctOfColumn']

N = len(y1)
ind = np.arange(N)
width = 0.25

red = '#F15854' #(red)
blue = '#5DA5DA' # (blue)
green = '#60BD68' # (green)

# NOTE: the main mechanism for side-by-side bar charts is adjusting the LEFT value by a width offset
p1 = plt.bar(left=ind - width, height=y1, color=red, width=width, align='center')
p2 = plt.bar(left=ind, height=y2, color=blue, width=width, align='center')
p3 = plt.bar(left=ind + width, height=y3, color=green, width=width, align='center')

plt.ylabel('y as % of x')
plt.xlabel('x values')
plt.title('distribution of y values within each x')
plt.xticks(ind, ('x1', 'x2', 'x3', 'All'))
plt.legend((p1[0], p2[0], p3[0]), ('y1', 'y2', 'y3'))
plt.show()

y1 = ctTemp.loc['y1', 'pctOfColumn']
y2 = ctTemp.loc['y2', 'pctOfColumn']
y3 = ctTemp.loc['y3', 'pctOfColumn']

N = len(y1)
ind = np.arange(N)
width = 0.35

red = '#F15854' #(red)
blue = '#5DA5DA' # (blue)
green = '#60BD68' # (green)

# NOTE: the main mechanism for segmented bar charts is adjusting the BOTTOM value by a cumulative height offset
p1 = plt.bar(ind, y1, color=red, width=width, align='center')
p2 = plt.bar(ind, y2, bottom=y1, color=blue, width=width, align='center')
p3 = plt.bar(ind, y3, bottom=y1 + y2, color=green, width=width, align='center')

plt.ylabel('y as % of x')
plt.xlabel('x values')
plt.title('distribution of y values within each x')
plt.xticks(ind, ('x1', 'x2', 'x3', 'All'))
plt.legend((p1[0], p2[0], p3[0]), ('y1', 'y2', 'y3'))
plt.show()




# ignore this setup line
get_ipython().magic('matplotlib inline')

from os.path import join
import csv
datafname = join('data', 'climate', 'nasa-gistemp-annual-mean.csv')
with open(datafname, 'r') as rf:
    datarows = list(csv.DictReader(rf))    

import matplotlib.pyplot as plt

# take a look at the data
print(datarows[0])

xvals = []
yvals = []

for d in datarows:
    xvals.append(d['year'])
    yvals.append(d['annual_mean'])

xvals = [d['year'] for d in datarows]
yvals = [d['annual_mean'] for d in datarows]

years = [int(d['year']) for d in datarows]
means = [float(d['annual_mean']) for d in datarows]

fig, ax = plt.subplots()
ax.bar(years, means)

colds = []
hots = []
for d in datarows:
    yr = int(d['year'])
    mean = float(d['annual_mean'])
    if mean < 0:
        colds.append((yr, mean))
    else:
        hots.append((yr, mean))

fig, ax = plt.subplots()
ax.bar([x[0] for x in colds], [x[1] for x in colds], color='blue')                
ax.bar([x[0] for x in hots], [x[1] for x in hots], color='orange')                                




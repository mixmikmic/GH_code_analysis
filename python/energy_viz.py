get_ipython().magic('matplotlib inline')
import os
import requests
import matplotlib
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

ENERGY = "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"

def download_data(url, path='data'):
    if not os.path.exists(path):
        os.mkdir(path)

    response = requests.get(url)
    name = os.path.basename(url)
    with open(os.path.join(path, name), 'wb') as f:
        f.write(response.content)

download_data(ENERGY)

energy   = pd.read_excel('data/ENB2012_data.xlsx', sep=",")

energy.head()

energy.columns = ['compactness','surface_area','wall_area','roof_area','height',                  'orientation','glazing_area','distribution','heating_load','cooling_load']

energy.describe()

# We can use the ggplot style with Matplotlib, which is a little bit nicer-looking than the standard style.
matplotlib.style.use('ggplot')

energy.plot(kind='area', stacked=False,figsize=[20,10])

energy.plot(kind='scatter', x='roof_area', y='cooling_load', c='surface_area',figsize=[20,10])

energy.plot(kind='scatter', x='wall_area', y='heating_load', s=energy['glazing_area']*500,figsize=[20,10])

energy.plot(kind='box',figsize=(20,10))

energy['compactness'].plot(kind='hist', alpha=0.5, figsize=(20,10))

energy.hist(figsize=(20,10))

energy['wall_area'].plot(kind='kde')

areas = energy[['glazing_area','roof_area','surface_area','wall_area']]
scatter_matrix(areas, alpha=0.2, figsize=(18,18), diagonal='kde')

x = [1, 2, 3, 4]
y = [1, 4, 9, 6]
labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs']

plt.plot(x, y, 'ro')
# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation=30)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.show()

sns.set_style('whitegrid')
sns.violinplot(x='height',y='cooling_load', data=energy)

sns.regplot(x='wall_area', y='cooling_load', data=energy, x_estimator=np.mean)

sns.set(style="ticks")

# Create a dataset with many short random walks
rs = np.random.RandomState(4)
pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
pos -= pos[:, 0, np.newaxis]
step = np.tile(range(5), 20)
walk = np.repeat(range(20), 5)
df = pd.DataFrame(np.c_[pos.flat, step, walk],
                  columns=["position", "step", "walk"])

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="walk", hue="walk", col_wrap=5, size=1.5)

# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0, ls=":", c=".5")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "step", "position", marker="o", ms=4)

# Adjust the tick positions and labels
grid.set(xticks=np.arange(5), yticks=[-3, 3],
         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)

sns.set()

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset('flights')
flights = flights_long.pivot('month', 'year', 'passengers')

# Draw a heatmap with the numeric values in each cell
sns.heatmap(flights, annot=True, fmt='d', linewidths=.5)

from string import ascii_letters as letters
sns.set(style="white")
 
# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(letters[:26]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


import matplotlib.pyplot as plt
# Jupyter magic to render all images in-line
get_ipython().magic('matplotlib inline')

import pandas as pd
import pandas.rpy.common as rcom
import pandas as pd
import numpy as np
iris = rcom.load_data('iris')
iris

plt.scatter(iris['Sepal.Width'], iris['Petal.Width'])

iris

# Set up a figure with 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scatter plot in top left
axes[0].scatter(iris['Sepal.Width'], iris['Petal.Width'])
axes[0].axis('off')

# Mean species petal widths in top right
means = iris.groupby('Species')['Petal.Width'].mean()
axes[1].bar(np.arange(len(means))+1, means)
# Note how broken this is without additional code
axes[1].set_xticklabels(means.index)

# More scatter plots, breaking up by species
colors = ['blue', 'green', 'red']
for i, (s, grp) in enumerate(iris.groupby('Species')):
    axes[2].scatter(grp['Sepal.Length'], grp['Petal.Length'], c=colors[i])

from skimage import data, io, filters

# The demo from http://scikit-image.org
image = data.coins()
image

plt.imshow(image, cmap='Greys_r');

edges = filters.sobel(image)
plt.imshow(edges, cmap='Greys_r');

# Demo taken from http://matplotlib.org/examples/lines_bars_and_markers/marker_fillstyle_reference.html

"""
Reference for marker fill-styles included with Matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


points = np.ones(5)  # Draw 3 points for each line
text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=12, fontdict={'family': 'monospace'})
marker_style = dict(color='cornflowerblue', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='gray')


def format_axes(ax):
    ax.margins(0.2)
    ax.set_axis_off()


def nice_repr(text):
    return repr(text).lstrip('u')


fig, ax = plt.subplots()

# Plot all fill styles.
for y, fill_style in enumerate(Line2D.fillStyles):
    ax.text(-0.5, y, nice_repr(fill_style), **text_style)
    ax.plot(y * points, fillstyle=fill_style, **marker_style)
    format_axes(ax)
    ax.set_title('fill style')

# Demo taken from http://matplotlib.org/examples/mplot3d/subplot3d_demo.html

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt


# imports specific to the plots in this example
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

# Twice as wide as it is tall.
fig = plt.figure(figsize=plt.figaspect(0.33))

#---- First subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim3d(-1.01, 1.01)

fig.colorbar(surf, shrink=0.5, aspect=10)

#---- Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')
X, Y, Z = get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10);

from nilearn.plotting import plot_stat_map
import nibabel as nb

# Create two images
img1 = nb.load('/Users/tal/Dropbox/Neurosynth/Results/Analysis2_reward_neurosynth/k2/reward_neurosynth_cluster_labels_PCA=100_k=2.nii.gz')
img2 = nb.load('/Users/tal/Dropbox/Neurosynth/Results/Analysis2_social_neurosynth/k2/social_neurosynth_cluster_labels_PCA=100_k=2.nii.gz')

# Set up the figure
fig, axes = plt.subplots(2, 1, figsize=(15, 4))

# Plot the two images--notice we pass a matplotlib Axes instance to each one!
p = plot_stat_map(img1, cut_coords=12, display_mode='z', title='Image 1', axes=axes[0], vmax=3)
plot_stat_map(img2, cut_coords=p.cut_coords, display_mode='z', title='Image 2', axes=axes[1], vmax=3)

# We can adjust mpl figure and axes properties after the fact
fig.suptitle("Look at the lovely brain images! LOOK AT THEM", fontsize=30);

# KDE plot of all iris attributes, collapsing over species
iris.plot(kind='kde')

# Separate boxplot of iris attributes for each species
iris.groupby('Species').boxplot(rot=45);

# From https://stanford.edu/~mwaskom/software/seaborn/examples/grouped_boxplot.html

import seaborn as sns
sns.set(style="ticks")

# Load the example tips dataset
tips = sns.load_dataset("tips")

print(tips.head())

# Draw a nested boxplot to show bills by day and sex
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")
sns.despine(offset=10, trim=True, )

# From https://stanford.edu/~mwaskom/software/seaborn/examples/scatterplot_matrix.html

sns.set()

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")

# Plotting on data-aware grids
attend = sns.load_dataset("attention").query("subject <= 12")
g = sns.FacetGrid(attend, col="subject", col_wrap=4, size=2, ylim=(0, 10))
g.map(sns.pointplot, "solutions", "score", color=".3", ci=None);

# From https://github.com/yhat/ggplot/blob/master/docs/Gallery.ipynb
from ggplot import *

ggplot(diamonds, aes(x='price', color='clarity')) +     geom_density() +     scale_color_brewer(type='div', palette=7) +     facet_wrap('cut')

# Adapted from http://bokeh.pydata.org/en/latest/docs/gallery/iris.html

from bokeh.plotting import figure, show, output_notebook
from bokeh.sampledata.iris import flowers

output_notebook()

colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
colors = [colormap[x] for x in flowers['species']]

p = figure(title = "Iris Morphology")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'

p.circle(flowers["petal_length"], flowers["petal_width"],
         color=colors, fill_alpha=0.2, size=10)

show(p)

# Adapted from http://bokeh.pydata.org/en/latest/docs/gallery/boxplot_chart.html

from bokeh.charts import BoxPlot, show
from bokeh.sampledata.autompg import autompg as df

print(df.head())

# origin = the source of the data that makes up the autompg dataset
title = "MPG by Cylinders and Data Source, Colored by Cylinders"

# color by one dimension and label by two dimensions
# coloring by one of the columns visually groups them together
box_plot = BoxPlot(df, label=['cyl', 'origin'], values='mpg',
                   color='cyl', title=title)

show(box_plot)

import plotly
import plotly.plotly as py
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import graph_objs
plotly.offline.init_notebook_mode()

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(619517)
Nr = 250
y = np.random.randn(Nr)
gr = np.random.choice(list("ABCDE"), Nr)
norm_params = [(0, 1.2), (0.7, 1), (-0.5, 1.4), (0.3, 1), (0.8, 0.9)]

for i, letter in enumerate("ABCDE"):
    y[gr == letter] *= norm_params[i][1] + norm_params[i][0]
df = pd.DataFrame(dict(Score = y, Group = gr))

data_header = 'Score'
group_header = 'Group'

colors_dict = dict(A = 'rgb(25, 200, 120)',
                   B = '#aa6ff60',
                   C = (0.3, 0.7, 0.3),
                   D = 'rgb(175, 25, 122)',
                   E = 'rgb(255, 150, 226)')

fig = FF.create_violin(df, data_header='Score', group_header='Group',
                       colors=colors_dict, height=500, width=800,
                       use_colorscale=False)
plotly.offline.iplot(fig, filename='Violin Plots with Dictionary Colors')

# Vega-Lite example--from https://vega.github.io/vega-lite/
{
  "data": {"url": "data/seattle-temps.csv"},
  "mark": "bar",
  "encoding": {
    "x": {
      "timeUnit": "month",
      "field": "date",
      "type": "temporal",
      "axis": {"shortTimeLabels": true}
            
    },
    "y": {
      "aggregate": "mean",
      "field": "temp",
      "type": "quantitative"
    }
  }
}

# From https://github.com/ellisonbg/altair/blob/master/altair/notebooks/01-Index.ipynb
from altair import datasets, Chart

data = datasets.load_dataset('cars')
# print(data.head(5))

c = Chart(data).mark_circle().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
)

c # save the chart as a variable and display here

# From http://holoviews.org

import numpy as np
import holoviews as hv
hv.notebook_extension('matplotlib')
fractal = hv.Image(np.load('mandelbrot.npy'))

((fractal * hv.HLine(y=0)).hist() + fractal.sample(y=0))

get_ipython().run_cell_magic('opts', "Points [scaling_factor=50] Contours (color='w')", "dots = np.linspace(-0.45, 0.45, 19)\n\nhv.HoloMap({y: (fractal * hv.Points(fractal.sample([(i,y) for i in dots])) +\n                fractal.sample(y=y) +\n                hv.operation.threshold(fractal, level=np.percentile(fractal.sample(y=y).data, 90)) +\n                hv.operation.contours(fractal, levels=[np.percentile(fractal.sample(y=y).data, 60)]))\n            for y in np.linspace(-0.3, 0.3, 21)}, kdims=['Y']).collate().cols(2)")


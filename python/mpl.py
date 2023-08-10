import matplotlib.pyplot as plt
import numpy as np

from bokeh import mpl
from bokeh.plotting import output_notebook, show

output_notebook()

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x / 2)
z = np.cos(x / 4)

plt.title("Matplotlib Figure in Bokeh")
plt.plot(x, y, "r-", marker='o')
plt.plot(x, z, "g-x", linestyle="-.")

show(mpl.to_bokeh())

from matplotlib.collections import LineCollection
from bokeh import mpl

# In order to efficiently plot many lines in a single set of axes,
# add the lines all at once. Here is a simple example showing how it is done.

N = 50
x = np.arange(N)
# Here are many sets of y to plot vs x
ys = [x + i for i in x]

colors = ['#ff0000', '#008000', '#0000ff', '#00bfbf', '#bfbf00', '#bf00bf', '#000000']

line_segments = LineCollection([list(zip(x, y)) for y in ys], color=colors,
                                linewidth=(0.5, 1, 1.5, 2), linestyle='dashed')

ax = plt.axes()
ax.add_collection(line_segments)
ax.set_title('Line Collection with dashed colors')

show(mpl.to_bokeh())

from matplotlib.collections import PolyCollection

# Generate data. In this case, we'll make a bunch of center-points and generate
# verticies by subtracting random offsets from those center-points
numpoly, numverts = 100, 4
centers = 100 * (np.random.random((numpoly, 2)) - 0.5)
offsets = 10 * (np.random.random((numverts, numpoly, 2)) - 0.5)
verts = centers + offsets
verts = np.swapaxes(verts, 0, 1)

# In your case, "verts" might be something like:
# verts = zip(zip(lon1, lat1), zip(lon2, lat2), ...)
# If "data" in your case is a numpy array, there are cleaner ways to reorder
# things to suit.

facecolors = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'black']

edgecolors = ['cyan', 'yellow', 'magenta', 'black', 'red', 'green', 'blue']

widths = [5, 10, 20, 10, 5]


# Make the collection and add it to the plot.
col = PolyCollection(verts, facecolor=facecolors, edgecolor=edgecolors,
                     linewidth=widths, linestyle='--', alpha=0.5)

ax = plt.axes()
ax.add_collection(col)

plt.xlim([-60, 60])
plt.ylim([-60, 60])

plt.title("MPL-PolyCollection support in Bokeh")

show(mpl.to_bokeh())




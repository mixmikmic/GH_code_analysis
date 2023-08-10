from sklearn.externals import joblib
model = joblib.load('data/knn-model.pkl')
model

import gsd.hoomd
from pathlib import Path
from statdyn.analysis.order import relative_orientations

with gsd.hoomd.open('data/unknown/configuration.gsd') as trj:
    snap = trj[0]
orientations = relative_orientations(snap.configuration.box,
                           snap.particles.position,
                           snap.particles.orientation,
                           max_neighbours=6,
                           max_radius=3.5)
classes = model.predict(orientations)
classes

import pandas
get_ipython().run_line_magic('matplotlib', 'inline')

pandas.Series(classes).value_counts().plot(kind='bar')

import numpy as np
from bokeh.plotting import show, output_notebook, figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import Colorblind4 as palette
from statdyn.figures.configuration import plot_circles, snapshot2data

# Output the final figure to the jupyter notebook
output_notebook()

# The colours we will assign each class
class_colours = {
    'liq': palette[0],
    'p2': palette[1],
    'p2gg': palette[2],
    'pg': palette[3],
}

# Convert the class strings to a colour for plotting
coloured_classes = [class_colours[c] for c in classes]

# Convert the snapshot to format ready for plotting 
data = snapshot2data(snap) 
data['colour'] = np.tile(coloured_classes, 3)
data['label'] = np.tile(classes, 3)
source = ColumnDataSource(data)

# Create the figure
p = figure(aspect_scale=1, match_aspect=True, width=920, height=800, active_scroll='wheel_zoom')
p.circle('x', 'y', radius='radius', source=source, color='colour', legend='label')

show(p)


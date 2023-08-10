# Experimentation using a ColumnDataSource

from bokeh.plotting import figure, show, output_notebook
from bokeh.models.sources import ColumnDataSource

x_values = [0, 1, 2, 3, 4, 5, 6]
y_values = [0.0, 1.0, 1.4, 1.7, 2.0, 2.2, 2.4]

p = figure()

d = {'x': [1, 2, 3], 'y': [5, 6, 7]}

# Data could be a dictionary OR pandas DataFrame
source = ColumnDataSource(data=d)

p.line(x=source.data['x'], y=source.data['y'])

output_notebook()
show(p)


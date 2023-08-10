from bokeh.plotting import figure, show, output_notebook
import numpy as np
from random import choice

p = figure()

color = choice(['red', 'green', 'blue', 'purple', 'black', 'orange', 'grey'])
x_values = [0, 1, 2, 3, 4, 5, 6]
y_values = np.random.randint(1, 20, 7)

p.line(x_values, y_values, line_color=color)

output_notebook()
show(p)

from bokeh.plotting import output_file

output_file('myresults.html')
show(p)

# On my machine, when I create a new figure()
# and then attempt to show() the new figure...
# It ALSO attempts to write to the file myresults.html,
# overwriting my previous result.

h = figure()
show(h)


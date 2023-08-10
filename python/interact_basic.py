from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook

import numpy as np

x = np.linspace(0, 2*np.pi, 2000)
y = np.sin(x)

output_notebook()

source = ColumnDataSource(data=dict(x=x, y=y))

p = figure(title="simple line example", plot_height=300, plot_width=600)
p.line(x, y, color="#2222aa", line_width=3, source=source, name="foo")

def update(f, w=1, A=1, phi=0):
    if   f == "sin": func = np.sin
    elif f == "cos": func = np.cos
    elif f == "tan": func = np.tan
    source.data['y'] = A * func(w * x + phi)
    source.push_notebook()

show(p)

from IPython.html.widgets import interact
interact(update, f=["sin", "cos", "tan"], w=(0,100), A=(1,10), phi=(0, 10, 0.1))




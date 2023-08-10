import numpy as np
from bokeh.plotting import figure, show, output_notebook

output_notebook(url="http://localhost:5006/")

N = 80
x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)

p = figure()
p.line(x,y, color="tomato", line_width=2)
show(p)




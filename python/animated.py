import time
from numpy import pi, cos, sin, linspace, roll, zeros_like
from bokeh.plotting import figure, show, output_notebook, output_server, curdoc
from bokeh.client import push_session

# create a new client session to the server
session = push_session(curdoc())

N = 50 + 1
r_base = 8
theta = linspace(0, 2*pi, N)
r_x = linspace(0, 6*pi, N-1)
rmin = r_base - cos(r_x) - 1
rmax = r_base + sin(r_x) + 1
colors = [
    "FFFFCC", "#C7E9B4", "#7FCDBB", "#41B6C4", "#2C7FB8", 
    "#253494", "#2C7FB8", "#41B6C4", "#7FCDBB", "#C7E9B4"
] * 5
cx = cy = zeros_like(rmin)

# first call output_server so the notebook cells
# are loaded from the configured server
output_server()
# then configure the default output state to generate output in
# Jupyter/IPython notebook cells when show is called
output_notebook()

p = figure(x_range=[-11, 11], y_range=[-11, 11])

p.annular_wedge(cx, cy, rmin, rmax, theta[:-1], theta[1:],
                fill_color = colors, line_color="black", name="aw")

renderer = p.select(dict(name="aw"))[0]
ds = renderer.data_source
show(p)
while True:
    rmin = ds.data["inner_radius"]
    rmin = roll(rmin, 1)
    ds.data["inner_radius"] = rmin
    
    rmax = ds.data["outer_radius"]
    rmax = roll(rmax, -1)
    ds.data["outer_radius"] = rmax
    
    time.sleep(.10)




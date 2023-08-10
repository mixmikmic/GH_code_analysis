import ipyvolume as ipv
import numpy as np
x, y, z = np.random.random((3, 10000))
ipv.quickscatter(x, y, z, size=1, marker="sphere", color="green")

get_ipython().run_line_magic('pinfo2', 'ipv.quickscatter')

import ipyvolume
import ipyvolume as ipv
import vaex
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
import ipyvolume.bokeh
output_notebook()

ds = vaex.example()
N = 10000

ipv.figure()
quiver = ipv.quiver(ds.data.x[:N],  ds.data.y[:N],  ds.data.z[:N],
                    ds.data.vx[:N], ds.data.vy[:N], ds.data.vz[:N],
                    size=1, size_selected=5, color_selected="grey")
ipv.xyzlim(-30, 30)
tools = "wheel_zoom,box_zoom,box_select,lasso_select,help,reset,"
#p = figure(title="E Lz space", tools=tools, webgl=True, width=500, height=500)
p = figure(title="E Lz space", tools=tools, width=500, height=500)

r = p.circle(ds.data.Lz[:N], ds.data.E[:N],color="navy", alpha=0.2)
# A 'trick' from ipyvolume to link the selection (one way traffic atm)
ipyvolume.bokeh.link_data_source_selection_to_widget(r.data_source, quiver, 'selected')


ipv.show()
show(p)



import holoviews as hv
import numpy as np
hv.extension('bokeh')

data = np.random.rand(600, 600, 10)

ds = hv.Dataset((np.arange(10),
                 np.linspace(0., 1., 600),
                 np.linspace(0., 1., 600),
                 data),
                kdims=['time', 'y', 'x'],
                vdims=['z'])

ds

get_ipython().run_line_magic('opts', "Image(cmap='viridis')")
ds.to(hv.Image, ['x', 'y']).hist()




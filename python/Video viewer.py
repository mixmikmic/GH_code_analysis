fname = 'blinkers-salt-pepper-dynamic.tif'

import imageio
import numpy as np
import holoviews as hv
hv.extension('bokeh', 'matplotlib')

a = np.array(imageio.mimread(fname))

ds = hv.Dataset((*(np.arange(x) for x in a[::10].shape), a[::10].T),
                ['Time', 'x', 'y'], 'Fluorescence')
ds

get_ipython().run_line_magic('opts', "Image (cmap='viridis')")
ds.to(hv.Image, ['x', 'y'])


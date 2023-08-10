get_ipython().magic('matplotlib qt')
import numpy as np
from mayavi import mlab

# try one example, figure is created by default
mlab.test_molecule()

# clear the figure then load another example
mlab.clf()
mlab.test_flow_anim()

# create a new figure
mlab.figure('mesh_example', bgcolor=(0,0,0,))
mlab.test_surf()


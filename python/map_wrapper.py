get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import gdx
from cgetools.map import *

# Open a GDX file containing output from a C-REM run
f = gdx.File('a.gdx')

# Plot a map of a 1-D parameter
live_map(f, 'aya')

# Plot a map of a 2-D parameter
live_map(f, 'houem')


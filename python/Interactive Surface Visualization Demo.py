get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
import sys
from niwidgets import niwidget_surface
from niwidgets.exampledata import examplesurface, exampleoverlays
from pathlib import Path

my_widget = niwidget_surface.SurfaceWidget(examplesurface, exampleoverlays)

my_widget.surface_plotter(showZeroes=True)




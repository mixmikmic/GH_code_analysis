# Complete set of Python 3.6 imports used for these examples

# Standard modules
import io
import logging
import lzma
import multiprocessing
import os
import ssl
import sys
import time
import urllib.request
import zipfile

# Third-party modules
import fastparquet      # Needs python-snappy and llvmlite
import graphviz         # To visualize Dask graphs 
import numpy as np
import pandas as pd
import psutil           # Memory stats
import dask
import dask.dataframe as dd
import bokeh.io         # For Dask profile graphs
import seaborn as sns   # For colormaps

# Support multiple lines of output in each cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Don't wrap tables
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
pd.options.display.width = 300

# Show matplotlib and bokeh graphs inline in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
bokeh.io.output_notebook()

print(sys.version)
np.__version__, pd.__version__, dask.__version__


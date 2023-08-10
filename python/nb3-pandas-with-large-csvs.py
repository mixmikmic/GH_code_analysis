# Standard modules
import io
import logging
import lzma
import multiprocessing
import os
import ssl
import time
import urllib.request
import zipfile

# Third-party modules
import fastparquet      # Needs python-snappy
import graphviz         # To visualize Dask graphs 
import numpy as np
import pandas as pd
import psutil           # Memory stats
import dask
import dask.dataframe as dd

# Support multiple lines of output in each cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Don't wrap tables
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
pd.options.display.width = 300

# Show matplotlib graphs inline in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

np.__version__, pd.__version__, dask.__version__

def memory_usage():
    """String with current memory usage in MB. Requires `psutil` package."""
    pid = os.getpid()
    mem_bytes = psutil.Process(pid).memory_info().rss
    return "[Process %s uses %0.1fMB]" % (pid, mem_bytes/1024.0/1024.0)

memory_usage()

get_ipython().run_cell_magic('time', '', 'df = d.read_csv(\'flights-2016-01.xz\', nrows=4, dialect="excel")')

df.T

memory_usage()

get_ipython().run_cell_magic('time', '', 'df = pd.read_csv(\'flights-2016-01.xz\', dialect="excel")')

memory_usage()

df.info()

df.memory_usage(deep=True).sum() / 1024 / 1024 

import textwrap
print('\n'.join(textwrap.wrap(', '.join(df.columns), 60)))

get_ipython().run_cell_magic('time', '', 'def load_months(months):\n    dfs = [ \n        pd.read_csv(\'flights-%s.xz\' % month, dialect="excel")\n            for month in months \n          ]\n    return pd.concat(dfs)')

df = load_months(['2015-12','2016-01','2016-02'])

memory_usage()

df.info()

df.memory_usage(deep=True).sum() / 1024 / 1024 




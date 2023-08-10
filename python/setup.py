get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n.container {\n    width: 100%;\n}\n</style>')

# python standard library
import sys
import os
import operator
import itertools
import collections
import functools
import glob
import csv
import datetime
import bisect
import sqlite3
import subprocess
import random
import gc
import shutil
import shelve
import contextlib
import tempfile
import math

# plotting setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_context('paper')
sns.set_style('darkgrid')
# use seaborn defaults
rcParams = plt.rcParams
rcParams['savefig.jpeg_quality'] = 100

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'retina', 'png'}")

# general purpose third party packages
import numpy as np
nnz = np.count_nonzero
import scipy
import scipy.stats
import scipy.spatial.distance
import numexpr
import h5py
import tables
import bcolz
import dask
import dask.array as da
import pandas as pd
import IPython
from IPython.display import clear_output, display, HTML
import sklearn
import sklearn.decomposition
import sklearn.manifold
import petl as etl
etl.config.display_index_header = True
import humanize
from humanize import naturalsize, intcomma, intword
import zarr
from scipy.stats import entropy
import lmfit

import allel

sys.path.insert(0, '../agam-report-base/src/python')
from util import *
import zcache
import veff
import hapclust
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'rockies')

sys.path.insert(0, '../scripts')
from setup import *




import urllib.request
import os
import pathlib
DATASET_DIR = pathlib.Path('../data')
DATASET_FILEPATH = pathlib.Path('../data/kddcup.data_10_percent.gz')
if not DATASET_DIR.exists():
    os.mkdir(DATA_DIR)
    f = urllib.request.urlretrieve(
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
        DATASET_FILEPATH)

import dask
import dask.bag as db
# Progress Bar
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

DATASET_FILEPATH

raw_data = db.read_text(DATASET_FILEPATH)

raw_data

raw_data.count().compute()

raw_data.take(5)

a = range(100)
data = db.from_sequence(a)

data

data = db.from_sequence(a, npartitions=10)

data

data.count()

data.count().compute()

data.take(10)

get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'dask')


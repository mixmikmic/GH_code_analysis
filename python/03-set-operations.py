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

raw_data = db.read_text(DATASET_FILEPATH).repartition(4)
# count all 
raw_data_count = raw_data.count().compute()

normal_raw_data = raw_data.filter(lambda x: "normal." in x)

# count normal
normal_raw_data_count = normal_raw_data.count().compute()

attack_raw_data = raw_data.remove(lambda x: "normal." in x)

# count attacks
attack_raw_data_count = attack_raw_data.count().compute()

print("There are {} normal interactions and {} attacks, from a total of {} interactions".format(normal_raw_data_count,
                                        attack_raw_data_count, raw_data_count))

csv_data = raw_data.map(lambda x: x.split(","))
protocols = csv_data.map(lambda x: x[1]).distinct()

protocols.compute()

protocols.visualize()

services = csv_data.map(lambda x: x[2]).distinct()
services.compute()

product = protocols.product(services).compute()
print("There are {} combinations of protocol X service".format(len(product)))

get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'dask')


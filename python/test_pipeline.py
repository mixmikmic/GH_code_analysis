import sys

sys.path.append('../../code/')
import os
import json
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import networkx as nx

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

data_dir = '../../data/'

from make_clean_data import *

make_clean_case_metadata(data_dir)

start = time.time()
make_clean_edgelist(data_dir, overwrite=True)
end = time.time()
print 'took %d seconds' % (end - start)

make_clean_jurisdiction_file(data_dir)

make_jurisdiction_edgelist(data_dir)








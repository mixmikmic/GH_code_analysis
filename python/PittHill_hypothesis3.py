# some programmatic housekeeping
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.random.seed(215)
get_ipython().magic('matplotlib inline')

notebook = "PittHill_hypothesis3.ipynb" # replace with FILENAME
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(notebook), '..', 'data')))




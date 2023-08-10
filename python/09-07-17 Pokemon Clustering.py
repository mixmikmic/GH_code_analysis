# Importing Libraries
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime as dt
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import warnings
import seaborn as sns
import sklearn as sk

from pylab import rcParams

# Set plotting format
#plt.rcParams['figure.figsize'] = (30,20)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Disable notebook warnings
warnings.filterwarnings('ignore')


# Setting Dataframe format
pd.set_option('display.max_columns',1000)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
pd.set_option('precision',2)


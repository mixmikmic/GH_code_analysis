import random

random.random()

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('matplotlib inline')

from decompositionplots import explore_manifold
explore_manifold()


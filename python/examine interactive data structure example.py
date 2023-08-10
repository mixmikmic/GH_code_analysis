# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")
    
# import the examine function
from jp_gene_viz.examine import examine

# Create a "network pair" widget object.
from jp_gene_viz import paired_networks
paired_networks.load_javascript_support()

P = paired_networks.PairedNetworks()
P.load_networks("network.tsv", "network2.tsv")

examine(P, component_limit=6)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

s = pd.Series([1,3,5,np.nan,6,8])
s

examine(s)

examine(d.__class__)




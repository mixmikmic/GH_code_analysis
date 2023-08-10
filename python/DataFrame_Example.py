import numpy as np
import pandas as pd
from clustergrammer_widget import *

# generate random matrix
num_rows = 500
num_cols = 10
np.random.seed(seed=100)
mat = np.random.rand(num_rows, num_cols)

# make row and col labels
rows = range(num_rows)
cols = range(num_cols)
rows = [str(i) for i in rows]
cols = [str(i) for i in cols]

# make dataframe 
df = pd.DataFrame(data=mat, columns=cols, index=rows)

# initialize network object
net = Network(clustergrammer_widget)
# load dataframe
net.load_df(df)
# cluster using default parameters
net.cluster(enrichrgram=False)
# make the visualization
net.widget()

net.normalize(axis='col', norm_type='zscore', keep_orig=True)
net.cluster(enrichrgram=False)
net.widget()

net.filter_N_top('row', 20, 'sum')
net.cluster(enrichrgram=False)
net.widget()




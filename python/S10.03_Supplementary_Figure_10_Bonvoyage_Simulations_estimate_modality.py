import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import six

sns.set(style='ticks', context='talk', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})


import bonvoyage

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Figures in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set random seed
np.random.seed(sum(map(ord, 'bonvoyage')))


# Define folder to save figures
folder = 'pdf'
get_ipython().system('mkdir -p $folder')

data = pd.read_csv('data.csv', index_col=0)
data.head()

import anchor

estimator = anchor.BayesianModalities()

modalities = estimator.fit_predict(data)
modalities.head()

modalities.to_csv('modalities.csv')




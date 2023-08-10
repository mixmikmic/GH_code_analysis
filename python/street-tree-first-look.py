import pandas as pd
import os
import urllib

FILE_PATH = 'Street_Tree_List.csv'
DATA_SOURCE_URL = 'https://data.sfgov.org/api/views/tkzw-k3nq/rows.csv?accessType=DOWNLOAD'

# Load from interwebs
if not os.path.isfile(FILE_PATH):
    urllib.urlretrieve (DATA_SOURCE_URL, FILE_PATH)
    
df_trees = pd.read_csv(FILE_PATH)

from IPython.display import display

display(df_trees.columns)

import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (10, 6)

df_valid_cord = df_trees[df_trees.Latitude > 37.6]

# Look at secondary caretakers, looking specifically for FuF
df_valid_cord.groupby('qCareAssistant').plot(kind='scatter', x='Longitude', y='Latitude')

import seaborn as sns
sns.set(style="ticks")

# Pretty plot with seaborn to look at tree distributions
sns.jointplot(df_valid_cord.Longitude, df_valid_cord.Latitude, kind="hex", color="#4CB391", size=10)




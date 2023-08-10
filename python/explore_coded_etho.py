from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from spiketopics.helpers import *

get_ipython().magic('matplotlib inline')
sns.set_style('darkgrid')

ethofile = 'sql/etho.csv'
etho = pd.read_csv(ethofile)

# rename some columns
etho = etho.rename(columns={'movieId': 'movie', 'frameNumber': 'frame'})
etho.head()

# get rid of categories that are either identifiers or have non-binary entries
edf = etho.drop(['frameTime', 'ethoCameraLabel', 'ethoFaceLabel',
                'ethoGenitalsLabel', 'ethoForageLabel', 'ethoAggressionLabel',
                'ethoRoughCountLabel'], axis=1)

edf.head()

def plot_code_durations(arr):
    """
    Given an array, get distributions of durations of each coded value.
    """
    _, runlens, values = rle(arr)
    codes = list(np.unique(values))
    for c in codes:
        rlens = runlens[values == c]
        plt.hist(np.log(rlens).ravel(), bins=50, normed=True, alpha=0.25, label=str(c));

for col in edf:
    if col not in ['movie', 'frame']:
        plt.figure()
        plot_code_durations(edf[col].values)
        plt.title(col)




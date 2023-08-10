get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
sys.path.append("..")
from script.incremental_SAX import Incremental_SAX

def clean_outlier(column):
    mean_column = column.mean()
    std_column = column.std()
    column[column > mean_column+3*std_column] = mean_column+3*std_column
    column[column < mean_column-3*std_column] = mean_column-3*std_column
    return column

eeg = pd.read_csv("../data/EEG_Eye_State.arff",comment="@",header=None)
Y = eeg[eeg.columns[-1]]
X = eeg.drop(eeg.columns[-1],1)
X = X.apply(clean_outlier)

z_eeg = znormalization(X)
split_eeg = paa_transform(z_eeg, 1500)
sax_eeg = sax_transform(X, 1500, 30, True)

plt.plot(range(Y.size),Y)
plt.plot(range(Y.size),z_eeg[3])
plt.ylim([-1,1])

alphabet_sz=30
eeg_id = 2

percentils2 = np.percentile(split_eeg,np.linspace(1./alphabet_sz, 
                                          1-1./alphabet_sz, 
                                          alphabet_sz-1)*100)

percentils = norm.ppf(np.linspace(1./alphabet_sz, 
                                          1-1./alphabet_sz, 
                                          alphabet_sz-1))
plt.plot(range(split_eeg[:,eeg_id].size),split_eeg[:,eeg_id])
for percentil in percentils:
    plt.plot((0, split_eeg[:,eeg_id].size), (percentil, percentil), 'k:')
for percentil in percentils2:
    plt.plot((0, split_eeg[:,eeg_id].size), (percentil, percentil), 'r:')
plt.ylim([-2.5,2.5])

import seaborn as sns
#palette = sns.color_palette("hls", 30)
#sns.palplot(sns.color_palette("hls", 30))
palette = sns.color_palette("coolwarm", 30)
sns.palplot(sns.color_palette("coolwarm", 30))

step = 1./sax_eeg.shape[1]
for k,eeg_id in enumerate(range(sax_eeg.shape[1])):
    for i,val in enumerate(sax_eeg[:,eeg_id]):
        plt.plot((i,i), (k*step,(k+1)*step), color=palette[val])

step = 1./sax_eeg.shape[1]
for k,eeg_id in enumerate(range(sax_eeg.shape[1])):
    for i,val in enumerate(sax_eeg[:,eeg_id]):
        plt.plot((i,i), (k*step,(k+1)*step), color=palette[val])

plt.plot(range(Y.size),Y)




#!pip install pyedflib
#!pip install PyWavelets

import pyedflib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pywt
from PIL import Image
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')

chb01_21_seisure_df = pd.read_csv('chb01_21_seisure.csv')
chb01_21_seisure_df = chb01_21_seisure_df.drop(chb01_21_seisure_df.columns[[-1]], axis=1)
chb01_26_seisure_df = pd.read_csv('chb01_26_seisure.csv')
chb01_26_seisure_df = chb01_26_seisure_df.drop(chb01_26_seisure_df.columns[[-1]], axis=1)
chb01_37_normal_1_df = pd.read_csv('chb01_37_normal_1.csv')
chb01_37_normal_1_df = chb01_37_normal_1_df.drop(chb01_37_normal_1_df.columns[[-1]], axis=1)
chb01_37_normal_1_df = pd.read_csv('chb01_37_normal_1.csv')
chb01_37_normal_1_df = chb01_37_normal_1_df.drop(chb01_37_normal_1_df.columns[[-1]], axis=1)
chb01_38_normal_1_df = pd.read_csv('chb01_38_normal_1.csv')
chb01_38_normal_1_df = chb01_38_normal_1_df.drop(chb01_38_normal_1_df.columns[[-1]], axis=1)
print chb01_21_seisure_df.shape
chb01_21_seisure_df.head(n=2)

plt.figure(figsize=(4,4))
plt.scatter(chb01_21_seisure_df.iloc[0].values, chb01_21_seisure_df.iloc[1].values, label='0,1')
plt.scatter(chb01_21_seisure_df.iloc[0].values, chb01_21_seisure_df.iloc[2].values, label='0,1')
plt.scatter(chb01_21_seisure_df.iloc[0].values, chb01_21_seisure_df.iloc[3].values, label='0,1')
plt.scatter(chb01_21_seisure_df.iloc[1].values, chb01_21_seisure_df.iloc[2].values, label='0,1')

plt.xlabel('Corelation 1'); plt.ylabel('Corelation 2')
plt.legend(loc='best')
plt.grid()
plt.show()

# Compute matrix of correlation coefficients
corr_matrix_chb01_37_normal_1_df = np.corrcoef(chb01_37_normal_1_df.T)
corr_matrix_chb01_21_seisure_df = np.corrcoef(chb01_21_seisure_df.T)

# Display heat map 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.pcolor(corr_matrix_chb01_21_seisure_df)
ax2.pcolor(corr_matrix_chb01_37_normal_1_df)
ax1.set_title('Heatmap of correlation matrix seisures')
ax2.set_title('Heatmap of correlation matrix normal')

plt.show()

plt.figure(figsize=(12,8))
ts = np.arange(0, chb01_21_seisure_df.shape[0], 1)
max_pred = chb01_21_seisure_df.shape[1] -1
sep = 0
for i in (range(0, max_pred)):
    sep += 350
    cleaned_sep = chb01_21_seisure_df.ix[:, i].values + sep
    plt.plot(ts, cleaned_sep, label='0,1')
plt.xlabel('time series'); plt.ylabel('Brain segments seisure')
plt.legend(loc='best')
plt.grid()
plt.show()

plt.figure(figsize=(12,8))
ts = np.arange(0, chb01_26_seisure_df.shape[0], 1)
max_pred = chb01_26_seisure_df.shape[1] -1
sep = 0
for i in (range(0, max_pred)):
    sep += 350
    cleaned_sep = chb01_26_seisure_df.ix[:, i].values + sep
    plt.plot(ts, cleaned_sep, label='0,1')
plt.xlabel('time series'); plt.ylabel('Brain segments seisure')
plt.legend(loc='best')
plt.grid()
plt.show()

plt.figure(figsize=(12,8))
ts = np.arange(0, chb01_37_normal_1_df.shape[0], 1)
max_pred = chb01_37_normal_1_df.shape[1] -1
sep = 0
for i in (range(0, max_pred)):
    sep += 350
    cleaned_sep = chb01_37_normal_1_df.ix[:, i].values + sep
    plt.plot(ts, cleaned_sep, label='0,1')
plt.xlabel('time series'); plt.ylabel('Brain segments Normal')
plt.legend(loc='best')
plt.grid()
plt.show()

plt.figure(figsize=(12,8))
ts = np.arange(0, chb01_38_normal_1_df.shape[0], 1)
max_pred = chb01_38_normal_1_df.shape[1] -1
sep = 0
for i in (range(0, max_pred)):
    sep += 350
    cleaned_sep = chb01_38_normal_1_df.ix[:, i].values + sep
    plt.plot(ts, cleaned_sep, label='0,1')
plt.xlabel('time series'); plt.ylabel('Brain segments Normal')
plt.legend(loc='best')
plt.grid()
plt.show()




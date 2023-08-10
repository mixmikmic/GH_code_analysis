get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import os,sys

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

basepath = os.path.expanduser('~/Desktop/src/AllState_Claims_Severity/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(2016)

from data import *

# common script to load datasets
train, test, sample_sub = load_data()

print('Shape of the training dataset ', train.shape)

print('Shape of the test dataset ', test.shape)

train.loss.describe()

sns.kdeplot(train.loss);

train.boxplot(column='loss');

sns.kdeplot(train.loss.map(np.log))
plt.title('Log transformation of Loss');

train.loss.map(np.floor).value_counts().reset_index().plot.scatter(x='index', y='loss');

train.loss.map(np.floor).value_counts().sort_values(ascending=False)

print('Number of unique loss values in the training set ', train.loss.nunique())

sns.regplot(x='id', y='loss', data=train, fit_reg=False);

loss = train.loss

# create variable based on the frequency of the target variable
loss_freq = train.groupby(['loss'])['loss'].transform(lambda x: len(x))

# Bin the variable with 10 bins
bins   = np.linspace(loss_freq.min(), loss_freq.max(), 3)
labels = np.digitize(loss_freq, bins)

loss.reset_index().groupby(['loss']).size()

# frequency count of various labels
pd.Series(labels).value_counts()


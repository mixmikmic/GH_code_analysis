get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data/raw/'

# load files
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
words = pd.read_csv(os.path.join(DATA_DIR, 'words.csv'), encoding='ISO-8859-1')

train.head()

test.head()

users.head()

words.head()

sns.kdeplot(train.Rating);

train.groupby('Artist')['Rating'].mean().plot();

train.groupby('Track')['Rating'].mean().plot();

train.groupby('Time')['Rating'].mean().plot();






















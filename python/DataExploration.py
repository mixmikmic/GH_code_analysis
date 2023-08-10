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

# load files
train      = pd.read_csv(os.path.join(basepath, 'data/raw/train.csv'))
test       = pd.read_csv(os.path.join(basepath, 'data/raw/test.csv'))
sample_sub = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv'))

train.head()

cont_features = [col for col in train.columns[1:-1] if 'cont' in col]

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(16, 8), sharey=True)

ri = 0
ci = 0

for i in range(0, 14):
    axes[ri][ci].scatter(train[cont_features[i]], train.loss)
    axes[ri][ci].set_xlabel(cont_features[i])
    
    ci += 1
    
    if ci > 4:
        ri += 1
        ci = 0
        
plt.tight_layout();

plt.scatter(train.id, train.loss);

strange_data = train.loc[train.loss > 4e4]

def summarize_dataset(df, features):
    
    for feat in features:
        print('Feature Name: %s\n'%(feat))
        if df[feat].dtype == np.object:
            feature_counts = df[feat].value_counts()
            print(df[feat].value_counts() / feature_counts.sum())
        else:
            print(df[feat].describe())
        print('='*50 + '\n')

features = train.columns[1:-1]

summarize_dataset(train.loc[train.loss <= 4e4] , features[:10])

summarize_dataset(strange_data, features[:10])

summarize_dataset(train.loc[train.loss <= 4e4] , features[10:20])

summarize_dataset(strange_data, features[10:20])

summarize_dataset(train.loc[train.loss <= 4e4] , features[20:30])

summarize_dataset(strange_data, features[20:30])

summarize_dataset(train.loc[train.loss <= 4e4] , features[30:40])

summarize_dataset(strange_data, features[30:40])

summarize_dataset(train.loc[train.loss <= 4e4] , features[40:50])

summarize_dataset(strange_data, features[40:50])

summarize_dataset(train.loc[train.loss <= 4e4] , features[50:60])

summarize_dataset(strange_data, features[50:60])

summarize_dataset(train.loc[train.loss <= 4e4] , features[60:70])

summarize_dataset(strange_data, features[60:70])

summarize_dataset(train.loc[train.loss <= 4e4] , features[70:80])

summarize_dataset(strange_data, features[70:80])

summarize_dataset(train.loc[train.loss <= 4e4] , features[80:90])

summarize_dataset(strange_data, features[80:90])

summarize_dataset(train.loc[train.loss <= 4e4] , features[90:100])

summarize_dataset(strange_data, features[90:100])

summarize_dataset(train.loc[train.loss <= 4e4] , features[100:110])

summarize_dataset(strange_data, features[100:110])

summarize_dataset(train.loc[train.loss <= 4e4] , features[110:116])

summarize_dataset(strange_data, features[110:116])




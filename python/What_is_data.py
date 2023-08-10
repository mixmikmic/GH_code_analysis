get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import os,sys

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations

import warnings
warnings.filterwarnings('ignore')

basepath = os.path.expanduser('~/Desktop/src/AllState_Claims_Severity/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(2016)

from data import *

train, test, sample_sub = load_data()

train.columns

# concatenate train and test set
data = pd.concat((train, test))

# set id as the index of our concatenated dataframe
data = data.set_index('id')

# take the target variable to log domain
data['log_loss'] = data.loss.map(np.log)

def unique_values(data):
    features = data.columns[:-1]
    
    for feat in features:
        print('Feature: {}'.format(feat))
        print('Number of unique values in training set: {}'.format(data[:len(train)][feat].nunique()))
        print('Number of unique values in test set: {}'.format(data[len(train):][feat].nunique()))
        print('\n')

unique_values(data)

data['cont1'].describe()

sns.kdeplot(data.cont1)

sns.regplot(x='cont1', y='log_loss', data=data[:len(train)], fit_reg=False);

continuous_variables = [col for col in data.columns if 'cont' in col]
print('Number of continuous variables are: {}'.format(len(continuous_variables)))

continuous_variables

fig, ax = plt.subplots(nrows=7, ncols=2, figsize=(15, 10), sharey=True)

sns.regplot(x='cont1', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[0][0])
sns.regplot(x='cont2', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[0][1])
sns.regplot(x='cont3', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[1][0])
sns.regplot(x='cont4', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[1][1])
sns.regplot(x='cont5', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[2][0])
sns.regplot(x='cont6', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[2][1])
sns.regplot(x='cont7', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[3][0])
sns.regplot(x='cont8', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[3][1])
sns.regplot(x='cont9', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[4][0])
sns.regplot(x='cont10', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[4][1])
sns.regplot(x='cont11', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[5][0])
sns.regplot(x='cont12', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[5][1])
sns.regplot(x='cont13', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[6][0])
sns.regplot(x='cont14', y='log_loss', data=data[:len(train)], fit_reg=False, ax=ax[6][1]);

cat1_cat2 = data[:len(train)].pivot_table(index='cat1', columns='cat2', values='loss', aggfunc='sum', margins=False)

cat1_cat2 / cat1_cat2.sum(axis=0)

data[data.loss < 20000].boxplot(column='loss', by='cat1');

data[data.loss < 20000].boxplot(column='loss', by='cat2');

cat1_cat3 = data[:len(train)].pivot_table(index='cat1', columns='cat3', values='loss', aggfunc='sum', margins=False)

cat1_cat3 / cat1_cat3.sum(axis=0)

data[data.loss < 20000].boxplot(column='loss', by='cat3');

cat2_cat4 = data[:len(train)].pivot_table(index='cat2', columns='cat4', values='loss', aggfunc='sum', margins=False)

cat2_cat4 / cat2_cat4.sum(axis=0)

cat2_cat3 = data[:len(train)].pivot_table(index='cat2', columns='cat3', values='loss', aggfunc='sum', margins=False)
cat2_cat3 / cat2_cat3.sum(axis=0)

cat3_cat4 = data[:len(train)].pivot_table(index='cat3', columns='cat4', values='loss', aggfunc='sum', margins=False)
cat3_cat4 / cat3_cat4.sum(axis=0)

categorical_variables = [col for col in data.columns if 'cat' in col]
print('Number of categorical variables ', len(categorical_variables))

list(combinations(['A', 'B', 'C'], 2))

def level_freq(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for feat in categorical_columns:
        if feat in ['loss', 'id']:
            continue
        print('Feat: {}'.format(feat))
        print('Number of unique values {}'.format(df[feat].nunique()))
        print('Frequency count per level \n')
        print(df[feat].value_counts())
        print('-'*50 + '\n')

level_freq(train)

level_freq(test)

class CategoricalFeatureInteraction(object):
    def __init__(self, df, features):
        self.df       = df
        self.features = features
    
    def capture_interaction(self):
        for (feat1, feat2) in combinations(self.features, 2):
            feat1_feat2 = self.df.pivot_table(index=feat1, columns=feat2, values='loss', aggfunc='sum', margins=False, fill_value=0)
            
            print('Feature Interaction {0}-{1}:\n'.format(feat1, feat2))
            print('-'*50)
            print('\n')
            
            print(feat1_feat2 / feat1_feat2.sum(axis=0))

cat_interaction = CategoricalFeatureInteraction(data[:len(train)], categorical_variables[:10])
cat_interaction.capture_interaction()

data[data.loss < 4e4].boxplot(column='loss', by='cat80')




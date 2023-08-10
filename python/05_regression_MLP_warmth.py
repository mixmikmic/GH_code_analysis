import io
import requests
import time # for timestamps

import numpy as np
import pandas as pd
from ast import literal_eval # parsing hp after tuner

from reg_tuning import * # my helper functions

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# fix random seed for reproducibility
seed = 2302
np.random.seed(seed)

path = 'https://raw.githubusercontent.com/laufergall/ML_Speaker_Characteristics/master/data/generated_data/'

url = path + "feats_ratings_scores_train.csv"
s = requests.get(url).content
feats_ratings_scores_train = pd.read_csv(io.StringIO(s.decode('utf-8')))

url = path + "feats_ratings_scores_test.csv"
s = requests.get(url).content
feats_ratings_scores_test = pd.read_csv(io.StringIO(s.decode('utf-8')))

with open(r'..\data\generated_data\feats_names.txt') as f:
    feats_names = f.readlines()
feats_names = [x.strip().strip('\'') for x in feats_names] 

with open(r'..\data\generated_data\items_names.txt') as f:
    items_names = f.readlines()
items_names = [x.strip().strip('\'') for x in items_names] 

with open(r'..\data\generated_data\traits_names.txt') as f:
    traits_names = f.readlines()
traits_names = [x.strip().strip('\'') for x in traits_names] 

# select a trait
# perform this on a loop later
target_trait = traits_names[0]

# train/test partitions, features and labels
X = np.load(r'.\data_while_tuning\X_' + target_trait + '.npy')
y = np.load(r'.\data_while_tuning\y_' + target_trait + '.npy')
Xt = np.load(r'.\data_while_tuning\Xt_' + target_trait + '.npy')
yt = np.load(r'.\data_while_tuning\yt_' + target_trait + '.npy')

# A/B splits, features and labels
AX = np.load(r'.\data_while_tuning\AX_' + target_trait + '.npy')
BX = np.load(r'.\data_while_tuning\BX_' + target_trait + '.npy')
Ay = np.load(r'.\data_while_tuning\Ay_' + target_trait + '.npy')
By = np.load(r'.\data_while_tuning\By_' + target_trait + '.npy')

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

"""
MLP with KerasRegressor
"""

def create_model(optimizer = 'Adam', learn_rate=0.2, neurons=1, activation='relu', dropout_rate=0.0):

    model = Sequential()

    model.add(Dense(neurons,
                    activation=activation, 
                    input_dim=88))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

# 1st round (tuning epochs, batch_size, neurons, learn rate):
    
def get_KerasRegressor2tune():
    
    model = KerasRegressor(build_fn = create_model, verbose=0)
                        
    hp = dict(
        regressor__epochs = [25,50,75,100],
        regressor__batch_size = [5,10], 
        regressor__neurons = [40, 80, 160],
        regressor__learn_rate = np.arange(start=0.2, stop=1.0, step=0.05) 
        #regressor__activation = ['relu'], # ['softmax', 'softplus', 'sofsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        #regressor__dropout_rate = [0.5], # np.arange(start=0, stop=1, step=0.1)
        #regressor__optimizer = ['Adam'] #['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    )
    return 'KerasRegressor', model, hp

tuning, trained = hp_tuner(AX, BX, Ay, By, 
                           [get_KerasRegressor2tune], 
                           target_trait,
                           feats_names,
                           [88], # no feature selection
                           'random',
                           n_iter=10
                          )

# 2nd round (tuning activation and dropout_rate):
    
def get_KerasRegressor2tune():
    
    model = KerasRegressor(build_fn = create_model, verbose=0)
                        
    hp = dict(
        regressor__epochs = [75], #[25,50,75,100],
        regressor__batch_size = [5], # [5,10], 
        regressor__neurons = [160], #[40, 80, 160],
        regressor__learn_rate = [0.8], # np.arange(start=0.2, stop=1.0, step=0.05) 
        regressor__activation = ['relu','tanh','sigmoid','linear'], # ['softmax', 'softplus', 'sofsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        regressor__dropout_rate = np.arange(start=0, stop=1, step=0.2)
        #regressor__optimizer = ['Adam'] #['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    )
    return 'KerasRegressor', model, hp


tuning, trained = hp_tuner(AX, BX, Ay, By, 
                           [get_KerasRegressor2tune], 
                           target_trait,
                           feats_names,
                           [88], # no feature selection
                           'grid'
                          )


yt_pred = trained[0].predict(Xt)

# average of outputs that belong to the same speaker

test_scores = pd.DataFrame(data = feats_ratings_scores_test[[target_trait,'spkID']])
test_scores['pred'] = yt_pred

test_scores_avg = test_scores.groupby('spkID').mean()

myrmse = np.sqrt(mean_squared_error(test_scores[target_trait].as_matrix(), 
 test_scores['pred'].as_matrix()))

myrmse_avg = np.sqrt(mean_squared_error(test_scores_avg[target_trait].as_matrix(), 
 test_scores_avg['pred'].as_matrix()))

print('RMSE per instance on B: %0.2f' % tuning['best_accs'])   
print('RMSE per instance: %0.2f' % myrmse)
print('RMSE after averaging over speaker utterances: %0.2f' % myrmse_avg) 


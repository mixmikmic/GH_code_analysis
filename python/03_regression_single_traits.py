import io
import requests
import time # for timestamps

import numpy as np
import pandas as pd
from ast import literal_eval # parsing hp after tuner

from reg_tuning import * # my helper functions

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Standardize speech features  

dropcolumns = ['name','spkID','speaker_gender'] + items_names + traits_names

# learn transformation on training data
scaler = StandardScaler()
scaler.fit(feats_ratings_scores_train.drop(dropcolumns, axis=1))

# numpy n_instances x n_feats
feats_s_train = scaler.transform(feats_ratings_scores_train.drop(dropcolumns, axis=1))
feats_s_test = scaler.transform(feats_ratings_scores_test.drop(dropcolumns, axis=1)) 

from sklearn.svm import SVR

"""
Support Vector Machines with rbf kernel
"""
def get_SVRrbf2tune():
    
    model = SVR()
    hp = dict(
        regressor__C = np.logspace(1,3,num=3),
        regressor__kernel = ['rbf'], 
        regressor__gamma = np.logspace(-3,-1,num=3)
    )
    return 'SVRrbf', model, hp

from sklearn.dummy import DummyRegressor

def trainDummyRegressor(X, y, AX, BX, Ay, By):

    model = DummyRegressor(strategy='mean')
    model.fit(AX, Ay)
    By_pred = model.predict(BX)
    score_on_B = np.sqrt(mean_squared_error(By, By_pred))
    d = {
        'regressors_names': ['DummyRegressor'],
        'best_accs': score_on_B,
        'best_hps': '',
        'sel_feats': '',
        'sel_feats_i': ''
        }

    tuning = pd.DataFrame(data = d)
    trained = model.fit(X, y)

    return tuning, [trained]


def test_RMSE(tuning_all, trained_all, Xt, yt):
    # go through performace for all regressors

    # removing duplicates from tuning_all (same classifier tuned twice with different searchers)
    indexes = tuning_all['regressors_names'].drop_duplicates(keep='last').index.values

    # dataframe for summary of performances
    # performances = pd.DataFrame(tuning_all.loc[indexes,['regressors_names','best_accs']])

    for i in indexes:

        yt_pred = trained_all[i][0].predict(Xt)

        # average of outputs that belong to the same speaker

        test_scores = pd.DataFrame(data = feats_ratings_scores_test[[target_trait,'spkID']])
        test_scores['pred'] = yt_pred

        test_scores_avg = test_scores.groupby('spkID').mean()

        myrmse = np.sqrt(mean_squared_error(test_scores[target_trait].as_matrix(), 
                     test_scores['pred'].as_matrix()))

        myrmse_avg = np.sqrt(mean_squared_error(test_scores_avg[target_trait].as_matrix(), 
                     test_scores_avg['pred'].as_matrix()))

        print('%r -> RMSE per instance on B: %0.2f' % (tuning_all.loc[i,'regressors_names'], tuning_all.loc[i,'best_accs']))   
        print('%r -> RMSE per instance: %0.2f' % (tuning_all.loc[i,'regressors_names'], myrmse))   
        print('%r -> RMSE after averaging over speaker utterances: %0.2f' % (tuning_all.loc[i,'regressors_names'], myrmse_avg))   

for target_trait in traits_names:
     
    print('')
    print(target_trait)
    
    X = feats_s_train # (2700, 88)
    y = feats_ratings_scores_train[target_trait].as_matrix() # (2700,)

    Xt = feats_s_test # (891, 88)
    yt = feats_ratings_scores_test[target_trait].as_matrix() # (891,)

    # split train data into 80% and 20% subsets - with balance in gender
    AX, BX, Ay, By = train_test_split(X, y, test_size=0.20, 
                                      stratify = feats_ratings_scores_train['speaker_gender'], 
                                      random_state=2302)


    # append tuning results and models
    tuning_all = pd.DataFrame()
    trained_all = []

    # tune and train SVR with rbf kernel
    tuning_svr, trained_svr = hp_tuner(AX, BX, Ay, By, 
                               [get_SVRrbf2tune], 
                               target_trait,
                               feats_names,
                               np.arange(50, AX.shape[1]+1), # selectKBest
                               'random',
                               n_iter=20
                              )
    tuning_all = tuning_all.append(tuning_svr, ignore_index=True)
    trained_all.append(trained_svr)
    
    # "train" dummy regressor
    tuning_dummy, trained_dummy = trainDummyRegressor(X, y, AX, BX, Ay, By)
    tuning_all = tuning_all.append(tuning_dummy, ignore_index=True)
    trained_all.append(trained_dummy)
    
    test_RMSE(tuning_all, trained_all, Xt, yt)


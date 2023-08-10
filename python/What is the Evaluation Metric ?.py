get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import os,sys
import bisect
import random

from collections import Counter, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

basepath = os.path.expanduser('~/Desktop/src/AllState_Claims_Severity/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(2016)

from data import *

train, test, sample_sub = load_data()

# target variable
y = train.loss

# lets generate some random predictions and see their mean absolute error

class RandomPredictions(object):
    def __init__(self, seed, length, max_, min_):
        self.seed   = np.random.seed(seed)
        self.length = length
        self.max_   = max_
        self.min_   = min_
        
    def generate_predictions(self):
        return [np.random.uniform(self.max_, self.min_) for i in range(self.length)]

rp1 = RandomPredictions(2016, len(y), y.max(), y.min())
rp2 = RandomPredictions(2015, len(y), y.max(), y.min())

preds1 = rp1.generate_predictions()
preds2 = rp2.generate_predictions()

err1   = mean_absolute_error(y, preds1)
err2   = mean_absolute_error(y, preds2)

print('Mean absolute error for case 1: ', err1)
print('Mean absolute error for case 2: ', err2)

# lets plot to see how the random predictions perform in general
def analyze_random_predictions():
    seeds = np.arange(2000, 2020)
    errs  = []
    
    for seed in seeds:
        rp    = RandomPredictions(seed, len(y), y.max(), y.min())
        preds = rp.generate_predictions()
        err   = mean_absolute_error(y, preds)
        
        errs.append(err)
    
    plt.scatter(np.arange(len(seeds)), errs)
    plt.xlabel('Number of trials')
    plt.ylabel('MAE of uniform random predictions');

analyze_random_predictions()

# lets generate some random predictions and see their mean absolute error

class WeightedPredictions(object):
    def __init__(self, seed, loss):
        self.seed   = np.random.seed(seed)
        self.loss   = loss
        
    def generate_predictions(self):
        return [np.random.choice(self.loss) for i in range(len(self.loss))]

# lets plot to see how the weighted predictions perform in general

def analyze_weighted_predictions():
    seeds = np.arange(2000, 2020)
    errs  = []
    
    for seed in seeds:
        rp    = WeightedPredictions(seed, y)
        preds = rp.generate_predictions()
        err   = mean_absolute_error(y, preds)
        
        errs.append(err)
    
    plt.scatter(np.arange(len(seeds)), errs)
    plt.xlabel('Number of trials')
    plt.ylabel('MAE of weighted predictions');

analyze_weighted_predictions();




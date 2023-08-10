import sys
print(sys.version)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import time

import pandas as pd
import seaborn as sns

import sys
sys.path.append('../code/')

get_ipython().system(' which python')

from least_squares_sgd import LeastSquaresSGD
from kernel import RBFKernel, Fourier
from mnist_helpers import mnist_training, mnist_testing
from hyperparameter_explorer import HyperparameterExplorer

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=60, 
                           n_informative=60, n_redundant=0, n_repeated=0, 
                           n_classes=5, n_clusters_per_class=1, 
                           weights=None, flip_y=0.001, class_sep=1.0, 
                           hypercube=True, shift=0.0, scale=1.0, 
                           shuffle=True, random_state=None)
X_train, y_train = X[0:120], y[0:120]
X_test, y_test = X[120:], y[120:]

h = HyperparameterExplorer(X=X_train, y=y_train, score_name='(square loss)/N, training',
                           primary_hyperparameter='sigma',
                           classifier = LeastSquaresSGD,
                           use_prev_best_weights=True,
                           test_X=X_test, test_y=y_test)

model_kwargs_sweep = {'delta_percent':0.01, 'max_epochs':10, 
                      'batch_size':5, # change epochs back to 15
                      'assess_test_data_during_fitting':False,
                      'check_W_bar_fit_during_fitting':False,
                      'eta0_search_start':0.001,
                      'verbose':False}

mk = model_kwargs_sweep.copy()
mk['verbose'] = True
h.train_model(model_kwargs=mk, kernel_kwargs={})

# check_W_bar_fit_during_fitting

h.models[1].results.head()

h.models[1].verbose=False
h.models[1].run_longer(epochs = 3, delta_percent = 0.05)

first_sigma = h.models[1].kernel.sigma
first_sigma

h.models[1].plot_loss_and_eta()

h.models[1].epochs #__dict__ #epoch

h.models[1].plot_W_bar_update_variance()

h.train_model(model_kwargs = model_kwargs_sweep,
              kernel_kwargs = {'sigma':first_sigma*2})

h.models

h.best_results_across_hyperparams()

h.summary

h.plot_fits()

h.plot_fits(df=h.best_results_across_hyperparams())

h.models

h.summary

h.models

h.train_on_whole_training_set()

# len(self.last_n_weights)
# self.results.shape

h.final_model.max_epochs

p = h.final_model.plot_log_loss_normalized_and_eta()
#p

h.evaluate_test_data()

h.final_model.W_bar is None

h.final_model.check_W_bar_fit_during_fitting

h.final_model.results.head()

h.final_model.results.columns

plot_cols = h.final_model.results.columns[h.final_model.results.columns.str.contains("square loss\)/N")]
plot_cols

h.final_model.results[plot_cols].head()

h.final_model.plot_loss_of_both_W_arrays(loss='square')

h.final_model.plot_loss_of_both_W_arrays(loss='0/1')

cols = [c for c in h.final_model.results if "loss" in c]
#cols

h.final_model.results[cols].head()

h.final_model.results.head(3)


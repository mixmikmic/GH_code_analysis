import sys
print(sys.version)

from functools import partial
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import time

import pandas as pd
import seaborn as sns

import sys
sys.path.append('../code/')

from least_squares_sgd import LeastSquaresSGD
from rbf_kernel import RBFKernel
from mnist_helpers import mnist_training, mnist_testing
from hyperparameter_explorer import HyperparameterExplorer

X_train, y_train = mnist_training(shuffled=True)

X_test, y_test = mnist_testing(shuffled=False)

data_points = 1000 
X_train, y_train = X_train[0:data_points], y_train[0:data_points]

h = HyperparameterExplorer(X=X_train, y=y_train, score_name='square loss',
                           primary_hyperparameter='sigma',
                           classifier = LeastSquaresSGD,
                           use_prev_best_weights=True,
                           test_X=X_test, test_y=y_test)

h.train_model(model_kwargs={'delta_percent':0.01, 'max_epochs':200}, kernel_kwargs={})

h.models[1].sigma

h.models[1].plot_loss_and_eta()

h.models[1].plot_w_hat_history()

h.train_model(model_kwargs={'delta_percent':0.01, 'max_epochs':200},
              kernel_kwargs ={'sigma':1e5})

h.models[2].plot_loss_and_eta()

h.models[2].plot_w_hat_history()

assert False

h.summary

h.train_model(model_kwargs={'eta0':100, 'delta_percent':0.01, 'max_epochs':500},
              kernel_kwargs ={'sigma':1e3})

h.best('model')

h.best('score')

h.summary

h.plot_fits()

assert False

model = LeastSquaresSGD(X=X_train, y=y_train, batch_size=100, kernel=RBF,
                        eta0=1e3, verbose=True,
                        progress_monitoring_freq=2000, max_epochs=500)

model.eta0

model.run()

model.results

model.plot_01_loss()

model.plot_01_loss(logx=True)

model.plot_square_loss(logx=False)

fig, ax = plt.subplots(1, 1, figsize=(4,3))
plot_data = model.results[model.results['step'] > 1]
plot_x = 'step'
plot_y = 'training (0/1 loss)/N'
colors = ['gray']
plt.plot(plot_data[plot_x], plot_data[plot_y],
             linestyle='--', marker='o',
             color=colors[0])

model.plot_w_hat_history()

model.results.columns

model.plot_loss_and_eta()

model_diverge = LeastSquaresSGD(X=X, y=y, batch_size=2, eta0 = model.eta0*10,
                                kernel=RBFKernel, 
                                progress_monitoring_freq=100, max_epochs=500)
model_diverge.run()

model_diverge.plot_square_loss()


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

from least_squares_sgd import LeastSquaresSGD

from rbf_kernel import RBFKernel

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=60, 
                           n_informative=60, n_redundant=0, n_repeated=0, 
                           n_classes=5, n_clusters_per_class=1, 
                           weights=None, flip_y=0.001, class_sep=1.0, 
                           hypercube=True, shift=0.0, scale=1.0, 
                           shuffle=True, random_state=None)

model = LeastSquaresSGD(X=X, y=y, batch_size=10, kernel=RBFKernel,
                        progress_monitoring_freq=100, max_epochs=1000)

model.eta0

model.eta0 # = model.eta0/10

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


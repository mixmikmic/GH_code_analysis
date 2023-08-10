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

X_train, y_train = mnist_training()

data_points = 1000 
X_train, y_train = X_train[0:data_points], y_train[0:data_points]

model = LeastSquaresSGD(X=X_train, y=y_train, batch_size=100, kernel=RBFKernel,
                        verbose=True,
                        progress_monitoring_freq=2000, max_epochs=50)

model.eta0

model.run()

model.results

model.plot_01_loss()

model.plot_01_loss(logx=True)

model.plot_square_loss(logx=False)

model.plot_w_hat_history()

model.results.columns

model.plot_loss_and_eta()

model_diverge = LeastSquaresSGD(X=X, y=y, batch_size=2, eta0 = model.eta0*10,
                                kernel=RBFKernel, 
                                progress_monitoring_freq=100, max_epochs=500)
model_diverge.run()

model_diverge.plot_square_loss()


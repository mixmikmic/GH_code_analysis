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

from mnist_helpers import mnist_training, mnist_testing
from hyperparameter_explorer import HyperparameterExplorer
from least_squares_sgd import LeastSquaresSGD
from kernel import Fourier

X_train_untransformed, y_train = mnist_training(shuffled=False) 
X_train = np.load('../notebooks/data/X_transformed_by_50_components.npy')

X_test_untransformed, y_test = mnist_testing(shuffled=False)
X_test = np.load('../notebooks/data/X_test_transformed_by_50_components.npy')

ls60k = LeastSquaresSGD(X_train, 
                     y_train,
                     max_epochs=2,
                        eta0_search_start = 100,
                     eta0_max_pts=X_train.shape[0],
                     verbose=True,
                     assess_test_data_during_fitting = False)


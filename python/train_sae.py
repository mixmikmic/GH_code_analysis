import os
os.chdir('~/Codes/DL - Topic Modelling')

from __future__ import print_function, division
import sys
import timeit
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd

import theano
import theano.tensor as T

from lib.deeplearning import autoencoder

dat_x = np.genfromtxt('data/dtm_20news.csv', dtype='float32', delimiter=',', skip_header = 1)
dat_y = dat_x[:,0]
dat_x = dat_x[:,1:]
vocab =  np.genfromtxt('data/dtm_20news.csv', dtype=str, delimiter=',', max_rows = 1)[1:]
test_input = theano.shared(dat_x)

model = autoencoder( architecture = [2756, 500, 500, 128], opt_epochs = [900,5,10], model_src = 'params/dbn_params')

model.train(test_input, batch_size = 100, epochs = 110, add_noise = 16, output_path = 'params/to_delete')

model = autoencoder( architecture = [2000, 500, 500, 128], model_src = 'params_2000/ae_train',  param_type = 'ae')

output = model.score(test_input)

colnames = ['bit'] * 128
colnames = [colnames[i] + str(i) for i in range(128)]
colnames.insert(0,'_label_')
pd.DataFrame(data = np.c_[dat_y, output], 
             columns = colnames). \
             to_csv( 'data/ae_features.csv', index = False)

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt_dat = np.genfromtxt('params_2000/ae_train/cost_profile.csv', delimiter=',', names = True)
plt.plot(plt_dat)
plt.show()


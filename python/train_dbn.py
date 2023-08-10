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

from lib.deeplearning import deepbeliefnet

# loading the data and transforming it into theano compatible variables
dat_x = np.genfromtxt('data/dtm_20news.csv', dtype='float32', delimiter=',', skip_header = 1)
dat_y = dat_x[:,0]
dat_x = dat_x[:,1:]
vocab =  np.genfromtxt('data/dtm_20news.csv', dtype=str, delimiter=',', max_rows = 1)[1:]
x = theano.shared(dat_x)
y = T.cast(dat_y, dtype='int32')

model = deepbeliefnet(architecture = [2756, 500, 500, 128])
model.pretrain(input = x, pretraining_epochs = 10, output_path = 'params/to_delete')

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt_dat = np.genfromtxt('params_2000/dbn_params_pretrain/lproxy_layer_2.csv', delimiter=',', names = True)[:20]
plt.plot(plt_dat)
plt.show()

model = deepbeliefnet(architecture = [2756, 500, 500, 128], opt_epochs = [900,5,10],
                      predefined_weights = 'params/dbn_params')
output = model.score(input = x)

colnames = ['bit'] * 128
colnames = [colnames[i] + str(i) for i in range(128)]
colnames.insert(0,'_label_')
pd.DataFrame(data = np.c_[dat_y, output], 
             columns = colnames). \
             to_csv( 'data/dbn_features.csv', index = False)

model = deepbeliefnet(architecture = [2756, 500, 500, 128], opt_epochs = [900,5,10], n_outs = 20, predefined_weights = 'params/dbn_params')
#model.train(x=x, y=y,batch_size = 70, training_epochs = 10, output_path = 'params/to_delete')
model.train(x=x, y=y, training_epochs = 10000, learning_rate = (1/70)/2, batch_size = 120,
            drop_out = [0.2, .5, .5, .5], output_path = 'params/to_delete')

model = deepbeliefnet(architecture = [2756, 500, 500, 128], n_outs = 20, predefined_weights = 'params/to_delete/trained_dbn.pkl')

sum([1 for i, j in zip(model.predict(x), dat_y) if i == j])/len(dat_y)

model.predict(x,prob=True)


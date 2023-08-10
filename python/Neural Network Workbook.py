import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
import os
from sklearn import preprocessing, metrics
from importlib import reload
import sknn_jgd.mlp
from sklearn import tree  #for graphing random forest tree
import pickle
get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = (10, 6)
inline_rc = dict(mpl.rcParams)
import src.nnload as nnload
import src.nntrain as nntrain
import src.nnplot as nnplot
# Set script parameters
minlev = 0.2
rainonly = False
write_nn_to_netcdf = False
fig_dir = './figs/'
data_dir = './data/'

r_str= 'X-StandardScaler-qTindi_Y-SimpleY-qTindi_Ntrnex400000_r_50R_50R_mom0.9reg1e-07_Niter10000_v3'
training_file='./data/conv_testing_v3.pkl'
r_mlp_eval, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, dlev =         pickle.load(open('./data/regressors/' + r_str + '.pkl', 'rb'))
# Load the data from the training/testing/validation file
x_scl, ypred_scl, ytrue_scl, x_unscl, ypred_unscl, ytrue_unscl =     nnload.get_x_y_pred_true(r_str, training_file, minlev=min(lev),
                             noshallow=False, rainonly=False)

reload(nnplot)
figpath = './figs/' + r_str + '/'
nnplot.plot_sample_profiles(1, x_unscl, ytrue_unscl, ypred_unscl, lev, figpath,samp=32337)

from imp import reload
reload(nnplot)
nnplot.meta_compare_error_rate_v2()

reload(nnplot)
nnplot.meta_plot_regs()

reload(nnplot)
nnplot.plot_neural_fortran('/Users/jgdwyer/day1100h00.1xday.nc.0001',
                           'X-StandardScaler-qTindi_Y-SimpleY-qTindi_Ntrnex125000_r_50R_mom0.9reg1e-06_Niter3000_v3',
                           latind=None, timeind=99,
                        ensemble=True)




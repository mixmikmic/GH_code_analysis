try:
    import cPickle as pickle
except:
    import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import lasagne
from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet, TrainSplit
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from create_and_train import EarlyStopping
from constant import Y_COLUMNS
from plotting_utils import plot_one_learning_curve, pplot_one_output
from plotting_utils import hplot_one_output, pplot_one_output_full
from lignet_utils import gen_train_test, load_nets, calc_r_squared
from lignet_utils import print_r_squared, transform_pred_to_actual

get_ipython().magic('matplotlib inline')

x_train, x_test, y_train, y_test, x_scaler, y_scaler = gen_train_test()

nets = load_nets('trained_networks/final*')
full_net = load_nets('trained_networks/full*')

with open('learning_curve.pkl', 'rb') as pkl:
    (train_scores_mean, train_scores_std, valid_scores_mean,
     valid_scores_std, train_sizes) = pickle.load(pkl)

# Make a learning curve (for full_net) to show how much training data is needed
plt.figure()
plt.title('Learning Curve')
plt.xlabel("num_samples")
plt.ylabel("MSE")

plt.plot(train_sizes, -train_scores_mean, 'o-', label='train', color='r')
plt.fill_between(train_sizes, -train_scores_mean - train_scores_std,
                -train_scores_mean + train_scores_std, alpha=0.2, color='r')
plt.plot(train_sizes, -valid_scores_mean, 'o-', label='valid', color='g')
plt.fill_between(train_sizes, -valid_scores_mean - valid_scores_std,
                -valid_scores_mean + valid_scores_std, alpha=0.2, color='g')
plt.legend(loc=0)
plt.axis([0, 6000, 0, 1.8])
# plt.savefig('learning_curve.png')

plot_one_learning_curve('all', full_net)

for key in nets:
    plot_one_learning_curve(key, nets)

plt.close('all')

# Plot a single output measure
f, ax = plt.subplots(figsize=(9, 9))
pplot_one_output(ax, nets[0], x_train, y_train, x_test, y_test, 0, Y_COLUMNS[0])

# Parity plots for the output measures in the full network
output_list = nets.keys()
f, ax = plt.subplots(int(round(len(output_list)/2.0)), 2, sharex=False,
                     sharey=False, figsize=(15, 3.5*len(output_list)))
ax = ax.ravel()

net = full_net['all']
# get the same train/test split that was used in setting up the net
Xt, Xv, yt, yv = net.train_split(x_train, y_train, net)
# calculate the values predicted by the network
ytpred = net.predict(Xt)
yvpred = net.predict(Xv)
ytestpred = net.predict(x_test)

for key in output_list:
    pplot_one_output_full(ax, yt, yv, y_test, ytpred, yvpred, ytestpred, key)
    
ax[len(output_list)-1].set_xlabel('Predicted Value')
ax[len(output_list)-2].set_xlabel('Predicted Value')
# put y-labels on the left hand subplots
for i in range(0, len(output_list), 2):
    ax[i].set_ylabel('Actual Value')
f.subplots_adjust(hspace=0.1, wspace=0.1)

# Parity plots for all the individually trained networks
output_list = nets.keys()
f, ax = plt.subplots(int(round(len(output_list)/2.0)), 2, sharex=False,
                     sharey=False, figsize=(15, 3.5*len(output_list)))
ax = ax.ravel()

for key in output_list:
    title = Y_COLUMNS[key]
    net = nets[key]
    pplot_one_output(ax, net, x_train, y_train, x_test, y_test, key, title)
    
ax[len(output_list)-1].set_xlabel('Predicted Value')
ax[len(output_list)-2].set_xlabel('Predicted Value')
# put y-labels on the left hand subplots
for i in range(0, len(output_list), 2):
    ax[i].set_ylabel('Actual Value')
f.subplots_adjust(hspace=0.1, wspace=0.1)

solids = pd.DataFrame(np.stack((nets[0].predict(x_train).ravel(),
                                      y_train[:, 0]), axis=1))
solids.describe()

solids.hist()

ltars = pd.DataFrame(np.stack((nets[1].predict(x_train).ravel(),
                                      y_train[:, 1]), axis=1))
ltars.describe()

ltars.hist()

# R squared for single nets and full_net
r_squared = calc_r_squared(nets, x_train, y_train, x_test, y_test)
print_r_squared(r_squared)

r_squared_full = calc_r_squared(full_net, x_train, y_train, x_test, y_test)
print_r_squared(r_squared_full, 'Complete net')

# Plot a histogram for a single output measure
f, ax = plt.subplots(figsize=(8, 8))
col = 3
net = nets[col]
Xt, Xv, yt, yv = net.train_split(x_train, y_train[:, col], net)
ytpred = net.predict(Xt)
yvpred = net.predict(Xv)
ytestpred = net.predict(x_test)
hplot_one_output(yt, yv, y_test[:, col], ytpred, yvpred, ytestpred,
                 Y_COLUMNS[col], ax)

# Plot a histogram for the full network
f, ax = plt.subplots(figsize=(8, 8))
net = full_net['all']
Xt, Xv, yt, yv = net.train_split(x_train, y_train, net)
ytpred = net.predict(Xt)
yvpred = net.predict(Xv)
ytestpred = net.predict(x_test)
hplot_one_output(yt.flatten(), yv.flatten(), y_test.flatten(),
                 ytpred.flatten(), yvpred.flatten(), ytestpred.flatten(),
                 'Full', ax)

# Plot histograms for all the individually trained networks
# Note this cell takes a long time to evaluate
output_list = nets.keys()
f, ax = plt.subplots(int(round(len(output_list)/2.0)), 2,
                     figsize=(15, 3.5*len(output_list)))
ax = ax.ravel()

for key in output_list:
    net = nets[key]
    Xt, Xv, yt, yv = net.train_split(x_train, y_train[:, key], net)
    ytpred = net.predict(Xt)
    yvpred = net.predict(Xv)
    ytestpred = net.predict(x_test)
    hplot_one_output(yt, yv, y_test[:, key], ytpred, yvpred, ytestpred,
                     Y_COLUMNS[key], ax, key)
f.subplots_adjust(hspace=0.1, wspace=0.1)
    




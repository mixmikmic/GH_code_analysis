import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.learning_curve import validation_curve, learning_curve

from lignet_utils import gen_train_test, calc_r_squared_dt
from constant import Y_COLUMNS
from plotting_utils import pplot_one_output_full

get_ipython().magic('matplotlib inline')

x_train, x_test, y_train, y_test, x_scaler, y_scaler = gen_train_test()

# Make a validation curve for the max depth parameter
train_size = int(0.5 * x_train.shape[0])
xt = x_train[:train_size, :]
yt = y_train[:train_size, :]

param_range = range(5, 30)
train_scores, valid_scores = validation_curve(tree.DecisionTreeRegressor(
        min_samples_leaf=5),
                                              xt, yt, 'max_depth', param_range,
                                              scoring='mean_squared_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure()
plt.title('Validation Curve for max_depth')
plt.xlabel("max_depth")
plt.ylabel("-MSE")

plt.plot(param_range, -train_scores_mean, 'o-', label='train', color='r')
plt.fill_between(param_range, -train_scores_mean - train_scores_std,
                -train_scores_mean + train_scores_std, alpha=0.2, color='r')
plt.plot(param_range, -valid_scores_mean, 'o-', label='valid', color='g')
plt.fill_between(param_range, -valid_scores_mean - valid_scores_std,
                -valid_scores_mean + valid_scores_std, alpha=0.2, color='g')
plt.legend(loc=0)

# Make a learning curve to find out how much training data to use
train_size = int(1 * x_train.shape[0])
xt = x_train[:train_size, :]
yt = y_train[:train_size, :]

train_sizes, train_scores, valid_scores = learning_curve(
    tree.DecisionTreeRegressor(max_depth=15), xt, yt,
    train_sizes=[100, 500, 1500, 5000, 10000, 50000, 100000, 133333],
    scoring='mean_squared_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure()
plt.title('Learning Curve')
plt.xlabel("num_samples")
plt.ylabel("-MSE")

plt.plot(train_sizes, -train_scores_mean, 'o-', label='train', color='r')
plt.fill_between(train_sizes, -train_scores_mean - train_scores_std,
                -train_scores_mean + train_scores_std, alpha=0.2, color='r')
plt.plot(train_sizes, -valid_scores_mean, 'o-', label='valid', color='g')
plt.fill_between(train_sizes, -valid_scores_mean - valid_scores_std,
                -valid_scores_mean + valid_scores_std, alpha=0.2, color='g')
plt.legend(loc=0)

get_ipython().run_cell_magic('timeit', '', 'dtr_full = tree.DecisionTreeRegressor(max_depth=26, min_samples_leaf=2)\ndtr_full = dtr_full.fit(x_train, y_train)')

calc_r_squared_dt(dtr_full, x_train, x_test, y_train, y_test)

# Parity plots for the output measures in the full tree
# note there is no validation set so the test set is given twice
output_list = Y_COLUMNS
f, ax = plt.subplots(int(round(len(output_list)/2.0)), 2, sharex=False,
                     sharey=False, figsize=(15, 3.5*len(output_list)))
ax = ax.ravel()

ytpred = dtr_full.predict(x_train)
ytestpred = dtr_full.predict(x_test)
yvpred = ytestpred

for key, name in enumerate(output_list):
    pplot_one_output_full(ax, y_train, y_test, y_test,
                          ytpred, yvpred, ytestpred, key)
    
ax[len(output_list)-1].set_xlabel('Predicted Value')
ax[len(output_list)-2].set_xlabel('Predicted Value')
# put y-labels on the left hand subplots
for i in range(0, len(output_list), 2):
    ax[i].set_ylabel('Actual Value')
f.subplots_adjust(hspace=0.1, wspace=0.1)


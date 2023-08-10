import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split

get_ipython().magic('matplotlib inline')

N, K = (1000, 2)
pi = 3.14159

X = np.random.random(size=(N, K))
y = (5*X[:, 0] + np.sin(5*2*pi*X[:, 0])
     - 5*X[:, 1] - np.cos(5*2*pi*X[:, 1])
     #+ 0.01*np.sin(5*2*pi*X[:, 0])*np.cos(5*2*pi*X[:, 1])
     + np.random.normal(loc=0.0, scale=0.01, size=N))

X_train, X_test, y_train, y_test = train_test_split(X, y)

plt.plot(X_train[:, 0], y_train, 'o', alpha = 0.5)

plt.plot(X_train[:, 1], y_train, 'o', alpha = 0.5)

params = {
    'max_depth': 2,
    'eta': 0.1,
    'silent': 1,
    'eval_metric': 'rmse',
    'seed': 154
}

get_ipython().run_cell_magic('capture', '', "\ndtrain = xgb.DMatrix(X_train[:, [0]], label = y_train)\ndvalid = xgb.DMatrix(X_test[:, [0]], label = y_test)\n\nevallist  = [(dtrain, 'train'), (dvalid, 'eval')]\nmodel_no_constraints = xgb.train(params, dtrain, \n                                 num_boost_round = 1000, evals = evallist, \n                                 early_stopping_rounds = 10)")

def plot_one_feature_prediction(bst, X, y, idx=1, title=""):
    """For one-feature model, plot data and prediction."""
    
    x_scan = np.linspace(0, 1, 100)
    x_plot = xgb.DMatrix(x_scan.reshape((len(x_scan),1)))
    y_plot = bst.predict(x_plot, ntree_limit = bst.best_ntree_limit)

    plt.plot(x_scan, y_plot, color = 'black')
    plt.plot(X, y, 'o', alpha = 0.25)

plot_one_feature_prediction(model_no_constraints, X_test[:, 0], y_test, "")

get_ipython().run_cell_magic('capture', '', '\nparams_constrained = params.copy()\nparams_constrained[\'updater\'] = "grow_monotone_colmaker,prune"\nparams_constrained[\'monotone_constraints\'] = "(1)"\n\nevallist  = [(dtrain, \'train\'), (dvalid, \'eval\')]\nmodel_with_constraints = xgb.train(params_constrained, dtrain, \n                                 num_boost_round = 1000, evals = evallist, \n                                 early_stopping_rounds = 10)')

plot_one_feature_prediction(model_with_constraints, X_test[:, 0], y_test, "")

get_ipython().run_cell_magic('capture', '', "\ndtrain = xgb.DMatrix(X_train[:, [1]], label = y_train)\ndvalid = xgb.DMatrix(X_test[:, [1]], label = y_test)\n\nevallist  = [(dtrain, 'train'), (dvalid, 'eval')]\nmodel_no_constraints = xgb.train(params, dtrain, \n                                 num_boost_round = 1000, evals = evallist, \n                                 early_stopping_rounds = 10)")

plot_one_feature_prediction(model_no_constraints, X_test[:, 1], y_test, "")

get_ipython().run_cell_magic('capture', '', '\nparams_constrained = params.copy()\nparams_constrained[\'updater\'] = "grow_monotone_colmaker,prune"\nparams_constrained[\'monotone_constraints\'] = "(-1)"\n\nevallist  = [(dtrain, \'train\'), (dvalid, \'eval\')]\nmodel_with_constraints = xgb.train(params_constrained, dtrain, \n                                 num_boost_round = 1000, evals = evallist, \n                                 early_stopping_rounds = 10)')

plot_one_feature_prediction(model_with_constraints, X_test[:, 1], y_test, "")

get_ipython().run_cell_magic('capture', '', "\ndtrain = xgb.DMatrix(X_train, label = y_train)\ndvalid = xgb.DMatrix(X_test, label = y_test)\n\nevallist  = [(dtrain, 'train'), (dvalid, 'eval')]\nmodel_no_constraints = xgb.train(params, dtrain, \n                                 num_boost_round = 2500, evals = evallist, \n                                 early_stopping_rounds = 10)")

def plot_one_feature_of_two_prediction(bst, X, y, idx=1, title=""):
    """For one-feature model, plot data and prediction."""
    
    x_scan = np.linspace(0, 1, 100)    
    X_scan = np.empty((100, X.shape[1]))
    X_scan[:, idx] = x_scan
    
    left_feature_means = np.tile(X[:, :idx].mean(axis=0), (100, 1))
    right_feature_means = np.tile(X[:, (idx+1):].mean(axis=0), (100, 1))
    X_scan[:, :idx] = left_feature_means
    X_scan[:, (idx+1):] = right_feature_means
    
    
    X_plot = xgb.DMatrix(X_scan)
    y_plot = bst.predict(X_plot, ntree_limit=bst.best_ntree_limit)

    plt.plot(x_scan, y_plot, color = 'black')
    plt.plot(X[:, idx], y, 'o', alpha = 0.25)

plot_one_feature_of_two_prediction(model_no_constraints, X_test, y_test, 0)

plot_one_feature_of_two_prediction(model_no_constraints, X_test, y_test, 1)

get_ipython().run_cell_magic('capture', '', '\nparams_constrained = params.copy()\nparams_constrained[\'updater\'] = "grow_monotone_colmaker,prune"\nparams_constrained[\'monotone_constraints\'] = "(1,-1)"\n\nevallist  = [(dtrain, \'train\'), (dvalid, \'eval\')]\nmodel_with_constraints = xgb.train(params_constrained, dtrain, \n                                   num_boost_round = 2500, evals = evallist, \n                                   early_stopping_rounds = 10)')

plot_one_feature_of_two_prediction(model_with_constraints, X_test, y_test, 0)

plot_one_feature_of_two_prediction(model_with_constraints, X_test, y_test, 1)

get_ipython().run_cell_magic('timeit', '-n 100', 'model_no_constraints = xgb.train(params, dtrain, \n                                 num_boost_round = 2500, \n                                 verbose_eval = False)')

get_ipython().run_cell_magic('timeit', '-n 100', 'model_with_constraints = xgb.train(params_constrained, dtrain, \n                                 num_boost_round = 2500, \n                                 verbose_eval = False)')

from sklearn.datasets.california_housing import fetch_california_housing
cal_housing = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                                cal_housing.target,
                                                                test_size=0.5,
                                                                random_state=154)

dtrain = xgb.DMatrix(X_train, label = y_train)
dvalid = xgb.DMatrix(X_test, label = y_test)

def scatter_plot_feature(data, feature_names, feature_name, y):
    fid = feature_names.index(feature_name)
    x = data[:, fid].flatten()
    plt.plot(x, y, 'o', alpha=0.1)
    plt.title(feature_name)

scatter_plot_feature(X_train, cal_housing.feature_names, 'MedInc', y_train) 

scatter_plot_feature(X_train, cal_housing.feature_names, 'HouseAge', y_train)

scatter_plot_feature(X_train, cal_housing.feature_names, 'AveRooms', y_train)

scatter_plot_feature(X_train, cal_housing.feature_names, 'AveBedrms', y_train)

scatter_plot_feature(X_train, cal_housing.feature_names, 'Population', y_train) 

cal_housing.feature_names

get_ipython().run_cell_magic('timeit', '-n 10', 'model_no_constraints = xgb.train(params, dtrain, \n                                 num_boost_round = 2500, \n                                 verbose_eval = False)')

def make_constraint_spec(feature_names, constraint_dict):
    spec_list = []
    for fn in feature_names:
        if fn in constraint_dict:
            spec_list.append(str(constraint_dict[fn]))
        else:
            spec_list.append('0')
    return '(' + ','.join(spec_list) + ')'

params_constrained = params.copy()
params_constrained['updater'] = "grow_monotone_colmaker,prune"

monotone_constraints = {
    'MedInc': 1, 'HouseAge': 1, 'AveRooms': 1, 'AveBedrooms': 1, 'AveOccup': 1
}
params_constrained['monotone_constraints'] = make_constraint_spec(cal_housing.feature_names, monotone_constraints)

print(params_constrained['monotone_constraints'])

get_ipython().run_cell_magic('timeit', '-n 10', 'model_no_constraints = xgb.train(params, dtrain, \n                                 num_boost_round = 2500, \n                                 verbose_eval = False)')


from IPython.core.display import HTML
def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

ndata = 50

train_data = np.random.rand(2, ndata)
train_classes = np.where(((train_data[0,:]>0.4) & (train_data[1,:]>0.4)),1,-1)

test_data = np.random.rand(2,ndata)
test_classes = np.where(((test_data[0,:]>0.4) & (test_data[1,:]>0.4)),1,-1)

plt.plot(train_data[0, np.where(train_classes==-1)], train_data[1, np.where(train_classes==-1)],'ro')
plt.plot(train_data[0, np.where(train_classes==1)], train_data[1, np.where(train_classes==1)],'bo')
plt.title('Training Data');

plt.plot(test_data[0, np.where(test_classes==-1)], test_data[1, np.where(test_classes==-1)],'ro')
plt.plot(test_data[0, np.where(test_classes==1)], test_data[1, np.where(test_classes==1)],'bo')
plt.title('Test Data');

def classify(data, true_classes, value):
    
    # Classify
    classes = np.where(data<value, -1, 1)
    
    # Misclassification
    wrong = np.where(true_classes!=classes, 1, 0)
    
    return classes, wrong

classify(train_data[0], train_classes, 0.1)

def train(data, true_classes, weights):
        
    error = np.zeros(10)
    
    for value in range(10):
        
        val = value/10.
        classes = np.where(data<val, -1, 1)
        error[value] = np.sum(weights[np.where(true_classes!=classes)])

    return np.argmin(error)/10.

train(train_data[1], train_classes, np.ones(len(train_data[0]), float)/len(train_data[0]))

np.random.seed(42)
def adaboost(x_train, y_train, x_test, y_test, M=20):
    
    # Number of variables and training points
    K, N = np.shape(x_train)
    
    classifier = np.zeros(M)
    alpha = np.zeros(M)
    var = np.zeros(M, int)
    
    # Initial model weights
    w = np.ones(N, float)/N

    train_errors = np.zeros(M)
    test_errors = np.zeros(M)
    
    pred_train = np.zeros((M, N))
    pred_test = np.zeros((M, N))
    
    for m in range(M):
        
        # Pick random variable to use as classifier
        var[m] = np.random.binomial(1, 0.5)
        
        # Run classifier on variable using training x_train
        classifier[m] = train(x_train[var[m]], y_train, w)
        
        # Classify training and test x_train
        train_classes, train_error = classify(x_train[var[m]], y_train, classifier[m])
        test_classes, test_error = classify(x_test[var[m]], y_test, classifier[m])

        weighted_error = np.sum(w*train_error)/w.sum()
            
        if m and (weighted_error==0 or weighted_error>=0.5):
            M=m
            print('exiting at',m)
            break

        alpha[m] = np.log((1-weighted_error)/weighted_error)
        
        # Update weights
        w *= np.exp(alpha[m]*train_error)
        w /= np.sum(w)

        train_classes = np.zeros((N, m))
        test_classes = np.zeros((N, m))
        for i in range(m):
            train_classes[:, i], _ = classify(x_train[var[i]], y_train, classifier[i])
            test_classes[:, i], _ = classify(x_test[var[i]], y_test, classifier[i])

        # Make prediction based on series of boosted weak learners
        for n in range(N):
            pred_train[m, n] = np.sum(alpha[:m]*train_classes[n])/alpha.sum()
            pred_test[m, n] = np.sum(alpha[:m]*test_classes[n])/alpha.sum() 
            
        # Classify based on the sum of the weighted classes
        pred_train_classes = np.where(pred_train[m]>0, 1, -1)
        pred_test_classes = np.where(pred_test[m]>0, 1, -1)
        
        train_errors[m] = sum(pred_train_classes!=y_train)
        test_errors[m] = sum(pred_test_classes!=y_test)
    
    plt.plot(np.arange(M), train_errors[:M]/N,'k-', np.arange(M), test_errors[:M]/N,'k--')
    plt.legend(('Training error','Test error'))
    plt.xlabel('Iterations')
    plt.ylabel('Error')

adaboost(train_data, train_classes, test_data, test_classes)

np.random.seed(13)

def true_model(x):
    return x * np.sin(x) + np.sin(2 * x)

def gen_data(n_samples=200):
    x = np.random.uniform(0, 10, size=n_samples)
    x.sort()
    y = true_model(x) + 0.75 * np.random.normal(size=n_samples)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    x_train, y_train = x[train_mask, np.newaxis], y[train_mask]
    x_test, y_test = x[~train_mask, np.newaxis], y[~train_mask]
    
    return x_train, x_test, y_train, y_test
    

X_train, X_test, y_train, y_test = gen_data(200)

x_plot = np.linspace(0, 10, 500)

def plot_data(alpha=0.6):
    
    plt.figure(figsize=(8, 5))
    gt = plt.plot(x_plot, true_model(x_plot), alpha=alpha, label='true function')

    # plot training and testing data
    plt.scatter(X_train, y_train, s=10, alpha=alpha, label='training data')
    plt.scatter(X_test, y_test, s=10, alpha=alpha, color='red', label='testing data')
    plt.xlim((0, 10))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend(loc='upper left')
    
plot_data()

from sklearn.tree import DecisionTreeRegressor

plot_data()
est = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]),
         label='RT max_depth=1', color='g', alpha=0.9, linewidth=2)
plt.legend(loc='upper left');

from sklearn.tree import DecisionTreeRegressor

plot_data()
est = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]),
         label='RT max_depth=1', color='g', alpha=0.9, linewidth=2)

est = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]),
         label='RT max_depth=3', color='g', alpha=0.7, linewidth=1)

plt.legend(loc='upper left');

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from itertools import islice

plot_data()

est = GradientBoostingRegressor(n_estimators=1000, max_depth=1, learning_rate=1.0)
est.fit(X_train, y_train)

ax = plt.gca()
first = True

# step over prediction as we added 20 more trees.
for pred in islice(est.staged_predict(x_plot[:, np.newaxis]), 0, 1000, 20):
    
    plt.plot(x_plot, pred, color='r', alpha=0.2)
    
    if first:
        ax.annotate('High bias - low variance', xy=(x_plot[x_plot.shape[0] // 2],
                                                    pred[x_plot.shape[0] // 2]),
                                                    xycoords='data',
                                                    xytext=(3, 4), textcoords='data',
                                                    arrowprops=dict(arrowstyle="->",
                                                                    connectionstyle="arc"))
        first = False

pred = est.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, pred, color='r', label='GBRT max_depth=1')
ax.annotate('Low bias - high variance', xy=(x_plot[x_plot.shape[0] // 2],
                                            pred[x_plot.shape[0] // 2]),
                                            xycoords='data', 
                                            xytext=(6.25, -6),
                                            textcoords='data', 
                                            arrowprops=dict(arrowstyle="->",
                                                            connectionstyle="arc"))
plt.legend(loc='upper left');

colors = '#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666'

n_estimators = len(est.estimators_)

def deviance_plot(est, X_test, y_test, ax=None, label='', train_color=colors[0], 
                  test_color=colors[1], alpha=1.0):

    test_dev = np.array([est.loss_(y_test, pred) for pred in est.staged_predict(X_test)])

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        
    ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label, 
             linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color, 
             label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.set_ylabel('Error')
    ax.set_xlabel('n_estimators')
    ax.set_ylim((0, 2))
    
    return test_dev, ax

test_dev, ax = deviance_plot(est, X_test, y_test)
ax.legend(loc='upper right')

# add some annotations
ax.annotate('Lowest test error', xy=(test_dev.argmin() + 1, test_dev.min() + 0.02), xycoords='data',
            xytext=(150, 1.0), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
            )

ann = ax.annotate('', xy=(800, test_dev[799]),  xycoords='data',
                  xytext=(800, est.train_score_[799]), textcoords='data',
                  arrowprops=dict(arrowstyle="<->"))
ax.text(810, 0.25, 'train-test gap');

colors = '#2ca25f', '#99d8c9', '#e34a33', '#fdbb84', '#c51b8a', '#fa9fb5'

def fmt_params(params):
    return ", ".join("{0}={1}".format(key, val) for key, val in params.items())

fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({}, (colors[0], colors[1])),
                                          ({'min_samples_leaf': 3},
                                           (colors[2], colors[3])),
                                         ({'min_samples_leaf': 10},
                                           (colors[4], colors[5]))]:
    est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0)
    est.set_params(**params)
    est.fit(X_train, y_train)
    
    test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
                                 train_color=train_color, test_color=test_color)
    
ax.annotate('Higher bias', xy=(900, est.train_score_[899]), xycoords='data',
            xytext=(600, 0.3), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
            )
ax.annotate('Lower variance', xy=(900, test_dev[899]), xycoords='data',
            xytext=(600, 0.4), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
            )
plt.legend(loc='upper right')

n_estimators = 6000

fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({}, (colors[0], colors[1])),
                                          ({'learning_rate': 0.1},
                                           (colors[2], colors[3])),
                                         ({'learning_rate': 0.01},
                                           (colors[4], colors[5]))]:
    est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0)
    est.set_params(**params)
    est.fit(X_train, y_train)
    
    test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
                                 train_color=train_color, test_color=test_color)
    
ax.annotate('Requires more trees', xy=(200, est.train_score_[199]), xycoords='data',
            xytext=(300, 1.0), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
            )
ax.annotate('Lower test error', xy=(900, test_dev[899]), xycoords='data',
            xytext=(600, 0.5), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
            )
plt.legend(loc='upper right')

fig = plt.figure(figsize=(8, 5))
ax = plt.gca()

for params, (test_color, train_color) in [({}, (colors[0], colors[1])),
                                          ({'learning_rate': 0.1, 'subsample': 0.5},
                                           (colors[2], colors[3]))]:
    est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0,
                                    random_state=1)
    est.set_params(**params)
    est.fit(X_train, y_train)
    test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
                                 train_color=train_color, test_color=test_color)
    
ax.annotate('Even lower test error', xy=(400, test_dev[399]), xycoords='data',
            xytext=(500, 0.5), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
            )

est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0,
                                subsample=0.5)
est.fit(X_train, y_train)
test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params({'subsample': 0.5}),
                             train_color=colors[4], test_color=colors[5], alpha=0.5)
ax.annotate('Subsample alone does poorly', xy=(300, test_dev[299]), xycoords='data',
            xytext=(250, 1.0), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
            )
plt.legend(loc='upper right', fontsize='small')

from sklearn.model_selection import GridSearchCV

param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9, 17],
              # 'max_features': [1.0, 0.3, 0.1] ## not possible in our example (only 1 fx)
              }

est = GradientBoostingRegressor(n_estimators=3000)
# this may take some minutes
gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(X_train, y_train)

# best hyperparameter setting
gs_cv.best_params_

import pandas as pd

from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split

cal_housing = fetch_california_housing()

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    np.log(cal_housing.target),
                                                    test_size=0.2,
                                                    random_state=1)
names = cal_housing.feature_names

cal_housing

import pandas as pd

X_df = pd.DataFrame(data=X_train, columns=names)
X_df['LogMedHouseVal'] = y_train
_ = X_df.hist(column=['Latitude', 'Longitude', 'MedInc', 'LogMedHouseVal'])

est = GradientBoostingRegressor(n_estimators=3000, max_depth=6, learning_rate=0.04,
                                loss='huber', random_state=0)
est.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, est.predict(X_test))
print('MAE: %.4f' % mae)

# sort importances
indices = np.argsort(est.feature_importances_)

# plot as bar chart
plt.barh(np.arange(len(names)), est.feature_importances_[indices])
plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
_ = plt.xlabel('Relative importance')

from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms',
            ('AveOccup', 'HouseAge'), ('MedInc', 'HouseAge')]
fig, axs = plot_partial_dependence(est, X_train, features,
                                   feature_names=names, figsize=(14, 10))

import pandas as pd

vlbw = pd.read_csv('../data/vlbw.csv', index_col=0)
vlbw.head()

# Write your answer here




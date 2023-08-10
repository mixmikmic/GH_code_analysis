from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from plotly import tools
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import plotly.graph_objs as go

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import graphviz
from sklearn import *

from copy import deepcopy
from scipy.stats.stats import pearsonr, spearmanr
from collections import Counter

import visualizer
import data_loader

df_loader = data_loader.df_loader()

loo = model_selection.LeaveOneOut()

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
categories = np.array(['bin','bowl','bucket','cup','jar','pottedplant','tire','vase']).reshape(-1,1)

df_survey = df_loader.load_survey()
df_filtered = df_loader.load_filterd('ci')
df_area = df_loader.load_area()
df_detect = df_loader.load_detect()

df_population = df_loader.load_population()
df_dengue_cases = df_loader.load_cases()

df_dengue_cases_2016 = pd.read_csv('DHF/dengue_caces_2016.csv') 
df_dengue_cases_2017 = pd.read_csv('DHF/dengue_caces_2017.csv') 



x_train, y_train = [], [] 
xs, ys = [], []

column = 'total'

mean_det, std_det = df_detect[column].mean(), df_detect[column].std()

mean_cases, std_cases = df_dengue_cases_2016['cases'].mean(), df_dengue_cases_2016['cases'].std()

subdist_list = df_dengue_cases_2016['subdist'].unique()
for subdist in subdist_list:
    detect = round(df_detect.loc[df_detect['subdist'] == subdist][column].mean(),2)
    area = round(df_area.loc[df_area['subdist'] == subdist]['area'].mean(),2)

    population = round(df_population.loc[df_population['subdist'] == subdist]['population'].mean(),2)
    n_villages = round(df_population.loc[df_population['subdist'] == subdist]['n_villages'].mean(),2)

    survey = round(df_filtered.loc[(df_filtered['subdist'] == subdist) 
#                                        & (df_filtered.index.month.isin([6,7,8,9,10,11]))
                                      ]['ci'].mean(), 2)
    
    cases = round(df_dengue_cases_2016.loc[(df_dengue_cases_2016['subdist'] == subdist)]['cases'].mean(), 2)

#     if np.isnan(survey): continue
    if np.isnan(detect) or np.isnan(cases) or np.isnan(population): continue
    if detect > mean_det+1*std_det or detect < mean_det-1*std_det: continue
    if cases > mean_cases+1*std_cases or cases < mean_cases-1*std_cases: continue
        
    formula = (population)
    
    ys.append(formula)
    xs.append(cases)
    
    x = df_detect.loc[df_detect['subdist'] == subdist].copy()
#     x = x[['bin','bowl','bucket','cup','jar','pottedplant','tire','vase']].copy()
#     x = x[['bin','bowl','bucket','cup','jar','pottedplant','tire']].copy()
    x = x[['bin','bowl','bucket','jar','pottedplant','tire']].copy()
#     x = x[['bucket','jar','pottedplant']].copy()

    month = df_detect.loc[df_detect['subdist'] == subdist].index.month[0]
    
    features = list(np.squeeze(x.values)) + [month, area, population]
#     features = list(np.squeeze(x.values)) + [area, population]

#     features = np.array(population)
    
    x_train.append(np.array(features))
    y_train.append(cases)
    

X = np.array(x_train)
y = np.array(y_train)
print('X_train.shape:', X.shape)

len(xs)
print('\nR-squared:', metrics.r2_score(xs, ys))
print('Person:', pearsonr(xs, ys))
print(spearmanr(xs, ys),'\n')

trace = go.Scatter(
    x = xs, 
    y = ys, 
    mode = 'markers', name='Subdistrict',
    marker = dict(size = 15, opacity = 0.4)
)

xs = np.array(xs)
ys = np.array(ys)

regr = linear_model.LinearRegression()
regr.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))

ys_pred = regr.predict(xs.reshape(-1, 1))
trace_2 = go.Scatter(
    x = xs, 
    y = np.squeeze(ys_pred), 
    mode = 'lines', name='Regression', line = dict(width = 4)
)

layout = dict(
    title = '121 Data points, Population<br>' + \
            'Pearson: 0.510, Spearman: 0.467',
    width=650, 
    xaxis = dict(title = 'Dengue cases'),
    yaxis = dict(title = 'Population'),
    font=dict(size=16)
)
iplot(go.Figure(data=[trace, trace_2], layout=layout))

regr.fit(ys.reshape(-1, 1), xs.reshape(-1, 1))
pred = np.squeeze(regr.predict(ys.reshape(-1, 1)))

print('\nR-squared:', metrics.r2_score(xs, pred))
print('Person:', pearsonr(xs, pred))
print(spearmanr(xs, pred),'\n')

parameter_grid_gb = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'max_features': [2, 3, 4, 5, 6, 7],
#     'subsample': [0.6, 0.8, 1],
    'learning_rate':[0.01, 0.05, 0.1]
}

parameter_grid_tree = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'max_features': [2, 3, 4, 5],
}

parameter_grid_svr = {
    'kernel': ['linear','poly','rbf'],
    'degree': [1,2,3,4,5,6]
}


parameter_grid_ada = {
    'base_estimator': [svr, dt],
    'n_estimators': [5, 10, 15, 20, 25],
    'loss': ['linear', 'square', 'exponential'],
    'learning_rate':[0.1]
}


# grid_search = model_selection.GridSearchCV(estimator=svm.SVR(), 
#                                            param_grid=parameter_grid_svr, 
#                                            cv=10,
#                                            n_jobs=8)

# grid_search = model_selection.GridSearchCV(estimator=ensemble.RandomForestRegressor(), 
#                                            param_grid=parameter_grid_tree, 
#                                            cv=20,
#                                            n_jobs=1)

# grid_search = model_selection.GridSearchCV(estimator=tree.DecisionTreeRegressor(), 
#                                            param_grid=parameter_grid_tree, 
#                                            cv=3,
#                                            n_jobs=1)

grid_search = model_selection.GridSearchCV(estimator=ensemble.GradientBoostingRegressor(), 
                                           param_grid=parameter_grid_gb, 
                                           cv=10,
                                           n_jobs=1)

_=grid_search.fit(X, y)
grid_search.best_score_, grid_search.best_params_

# _=grid_search.fit(X, y)
# grid_search.best_score_, grid_search.best_params_

# _=grid_search.fit(X, y)
# grid_search.best_score_, grid_search.best_params_

X = X.reshape(-1,1)

X[0], X.shape

svr = svm.SVR(kernel='poly',  degree=2)
rf = ensemble.RandomForestRegressor(max_depth=3, max_features=3)
dt = tree.DecisionTreeRegressor(max_depth=3, max_features=5)
gb = ensemble.GradientBoostingRegressor(learning_rate=0.01, max_depth=3, max_features=3, subsample=0.8)

linear = linear_model.LinearRegression()
bayes = linear_model.BayesianRidge()
knn = neighbors.KNeighborsRegressor()

ada = ensemble.AdaBoostRegressor()
ada_svr = ensemble.AdaBoostRegressor(svr, learning_rate=0.03, loss='linear')
ada_dt = ensemble.AdaBoostRegressor(dt, learning_rate=0.03, loss='linear')

regrs = [
    [linear, 'Linear Regression'],
#     [svm.NuSVR(kernel='poly', degree=3, tol=12.3, gamma=0.28), 'NuSVR'],
#     [svm.SVR(kernel='poly', degree=2, tol=0.1), 'SVR'],
#     [bayes, 'Bayesian Ridge'],
#     [rf, 'Random Forest'],
#     [dt, 'Decision Tree'],
#     [gb, 'Gradient Boosting'],
#     [ada_svr, 'Ada SVR'],
]

df_selection = []
for k in range(1):
    df_compare = []
    for regr, name in regrs:
        y_pred, y_true = [], []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            _=regr.fit(X_train, Y_train)
            pred = regr.predict(X_test)
            y_true.append(np.squeeze(Y_test))
            y_pred.append(np.squeeze(pred))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        df_compare.append([
            name+'-'+str(k+1),
            metrics.r2_score(y_true, y_pred),
            pearsonr(y_true, y_pred)[0],
            spearmanr(y_true, y_pred)[0]
        ])

    df_compare = pd.DataFrame.from_records(df_compare)
    df_compare.columns = ['Model','R-squared','Pearson','Spearman']
    df_compare = df_compare.set_index('Model')
    df_compare = df_compare.round(4)
    df_selection.append(df_compare)
df_selection = pd.concat(df_selection, axis=0)

tmp = pd.DataFrame([[df_selection['R-squared'].mean(), 
                     df_selection['Pearson'].mean(), 
                     df_selection['Spearman'].mean()]])
tmp.columns = ['R-squared','Pearson','Spearman']
tmp.index = ['Average']

df_selection = df_selection.append(tmp)
df_selection

visualizer.plot_correlation(
    regrs[0][0], 
    '121 data points: Linear Regression<br>',
    X, y, loo
)



categories = np.array(['bin','bowl','bucket','jar','pottedplant','tire',]).reshape(-1,1)

# features_name = np.concatenate((categories,[['month']]), axis=0)    
# features_name = np.concatenate((categories,[['month'], ['popluation']]), axis=0)    
features_name = np.concatenate((categories,[['month'], ['area'],['popluation']]), axis=0)    

# features_name = np.array([['bucket'], ['jar'], ['pottedplant']])
# features_name = np.array([['bucket'], ['jar'], ['pottedplant'], ['month']])
# features_name = np.array([['bucket'], ['jar'], ['pottedplant'], ['popluation']])

# features_name = np.array([['bucket'], ['jar'], ['pottedplant'], ['month'], ['popluation']])
# features_name = np.array([['bucket'], ['jar'], ['pottedplant'], ['month'], ['area'], ['popluation']])


# features_name = deepcopy(categories)
features_name
features_name.shape

visualizer.plot_importance(regrs[0][0], regrs[0][1], X, y, loo, features_name)






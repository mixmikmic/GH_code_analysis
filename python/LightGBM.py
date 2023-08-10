import pandas as pd
import xgboost as xgb
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib

from scipy.stats import skew
from scipy.stats.stats import pearsonr

get_ipython().magic("config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().magic('matplotlib inline')


DIR_PATH = './'

train = pd.read_csv("train.csv", encoding="ascii") 
macro = pd.read_csv("macro.csv", encoding="ascii")
test =  pd.read_csv("test.csv", encoding="ascii") 

all_data1 = pd.concat((train.loc[:,'timestamp':'price_doc'],
                      test.loc[:,'timestamp':'market_count_5000']))
#all_data1['unitprice'] = all_data1.price_doc/all_data1.full_sq

macro_imp = ['timestamp', 'oil_urals', 'cpi', 'usdrub', 'rts', 'mortgage_rate', 'balance_trade', 'brent', 'micex', 'micex_cbi_tr', 'micex_rgbi_tr', 'fixed_basket']
macro_usefeat = macro[macro_imp]

all_data = pd.merge(all_data1, macro_usefeat, on = 'timestamp', how='left')

#log transform the target:
#train['price_doc'] = np.log1p(train['price_doc'])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.5]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]
y = X_train.price_doc

print X_train.shape
print X_test.shape
print X_train.columns.values

for column in train.columns :
    if train[column].nunique() < 60 :
        print column, np.array(sorted(train[column].unique()))
    else :
        print column, train[column].nunique()

from sklearn.model_selection import train_test_split


train_local, validation = train_test_split(
    X_train, 
    test_size = 0.2, 
    random_state = 0
)

print X_train.shape
print train_local.shape

from sklearn.metrics import mean_squared_error

scores = []
#print validation.shape[0]
for C in np.linspace(4500000, 8500000, 300) :
    p = np.ones(validation.shape[0]) * C
    score = mean_squared_error(p, validation.price_doc)
    scores.append((score, C))

print 'Min error: %.2f, optimal constant prediction: %.2f' % min(scores)
print 'Mean target: %.2f, median target: %.2f' % (validation['price_doc'].mean(), validation['price_doc'].median())

features = np.array([column for column in X_train.columns if column != 'price_doc'])
features

import os
from pylightgbm.models import GBMRegressor
os.environ['LIGHTGBM_EXEC'] = "/Users/Lakshmi/LightGBM/lightgbm"

offset = 50

model = GBMRegressor(
    num_threads=-1,
    learning_rate = 0.03,
    num_iterations = 1000, # does no of iteration increase help? 
    verbose = False, 
    early_stopping_round = 50,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
)

model.fit(
    train_local[features].values, 
    train_local['price_doc'].values + offset, 
    test_data = [(
        validation[features].values, 
        validation['price_doc'].values + offset
    )]
    
)

p = model.predict(validation[features].values) - offset
mean_squared_error(p, validation['price_doc'])

scores = []
for prediction_multiplicator in np.linspace(0.95, 1.05, 101) :
    score = mean_squared_error(p * prediction_multiplicator, validation['price_doc'])
    scores.append((score, prediction_multiplicator))

M = min(scores)[1]
validation.loc[:, 'price_doc_lightgbm'] = p * M
print 'Min error: %.2f, optimal prediction multiplicator: %.3f' % min(scores)

model = GBMRegressor(
    num_threads=-1,
    learning_rate = 0.03,
    num_iterations = int(model.best_round / 0.9), 
    verbose = False, 
    early_stopping_round = 50,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
)

model.fit(train_local[features].values, (train_local['price_doc'].values + offset))

test.loc[:, 'price_doc_lightgbm'] = model.predict(X_test[features].values) - offset
test.loc[:, 'price_doc_lightgbm'] *= M

test.loc[:, 'price_doc'] = np.exp(test['price_doc_lightgbm'])

test[['id', 'price_doc']].to_csv('lightgbm.csv', index = False)
get_ipython().system('gzip -f lightgbm.csv')

import lightgbm as lgb

# create dataset for lightgbm
lgb_train = lgb.Dataset(train_local[features].values, train_local['price_doc'].values + offset)
lgb_test = lgb.Dataset(validation[features].values, validation['price_doc'].values + offset, reference=lgb_train)

# specify your configurations as a dict
params = {
    'num_leaves': 5,
    'metric': ('l1', 'l2'),
    'verbose': 0
}

import heapq

evals_result = {}  # to record eval results for plotting

print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train, lgb_test],
                feature_name=[train_local.dtypes.index[i] for i in range(1897)],
                #feature_name=['f' + str(i + 1) for i in range(451)],
                categorical_feature=[21],
                evals_result=evals_result,
                verbose_eval=10)
#print('Feature names:', gbm.feature_name())
feature_imp = np.array(gbm.feature_importance())
feature_cnt = 20
top_N_features = (-feature_imp).argsort()[:feature_cnt]
print('Feature importances:')
for i in range(feature_cnt):
    print train_local.dtypes.index[top_N_features[i]] 

print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='l1')
plt.show()

print('Plot feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=50, figsize=(28,30))

#print('Plot 84th tree...')  # one tree use categorical feature to split
#ax = lgb.plot_tree(gbm, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
#plt.show()

#print('Plot 84th tree with graphviz...')
#graph = lgb.create_tree_digraph(gbm, tree_index=83, name='Tree84')
#graph.render(view=True)

# save model to file
gbm.save_model('model.txt')

# load model to predict
print('Load model to predict')
bst = lgb.Booster(model_file='model.txt')
# can only predict with the best iteration (or the saving iteration)
y_pred = bst.predict(validation[features].values) - offset
# eval with loaded model
print validation[features].values.shape
print validation['price_doc'].values.shape

print('The rmse of loaded model\'s prediction is:', mean_squared_error(y_pred , validation['price_doc']))


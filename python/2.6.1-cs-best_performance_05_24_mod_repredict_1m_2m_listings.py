import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
from sklearn import model_selection, preprocessing
import xgboost as xgb
from datetime import timedelta
import datetime
import math
from sklearn import model_selection
# from sklearn import proprocessing
from sklearn import linear_model
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV

train_missing=pd.read_csv('./train_xgb.csv')
test_missing= pd.read_csv('./test_xgb.csv')

train_copy = train_missing

test_for_id_only = pd.read_csv('../Data/test_lonlat_id_CS_0521_feature.csv')

test_id = test_for_id_only.id

# test_id

# print train_missing.shape
# train_1m      = train_missing.loc[(train_missing.price_doc==1000000),:]
# train_missing = train_missing.loc[-(train_missing.price_doc==1000000),:]

# print train_missing.shape
# train_2m      = train_missing.loc[(train_missing.price_doc==1000000),:]
# train_missing = train_missing.loc[-(train_missing.price_doc==2000000),:]
# train_missing.shape

print train_missing.shape
train_1m_2m   = train_missing.loc[(train_missing.price_doc==1000000)|(train_missing.price_doc==2000000),:]
train_missing = train_missing.loc[-((train_missing.price_doc==1000000)|(train_missing.price_doc==2000000)),:]

print train_missing.shape
# train_2m      = train_missing.loc[(train_missing.price_doc==1000000),:]
# train_missing = train_missing.loc[-(train_missing.price_doc==2000000),:]
# train_missing.shape

train_missing['price_full_sq'] = (train_missing['price_doc']/train_missing['full_sq'].astype(float)).astype(int)
train_1m_2m['price_full_sq']   = (train_1m_2m['price_doc']/train_1m_2m['full_sq'].astype(float)).astype(int)
y_train_missing = train_missing['price_full_sq']

x_train_missing = train_missing.drop(['price_full_sq','price_doc','id','timestamp'],axis=1)
x_train_1m_2m   = train_1m_2m.drop(['price_full_sq','price_doc','id','timestamp'],axis=1)
x_test_missing  = test_missing.drop(['id','timestamp'],axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0
}



x_train_missing_subset = x_train_missing
x_train_1m_2m_subset   = x_train_1m_2m
x_test_missing_subset  = x_test_missing

dtrain_subset       = xgb.DMatrix(x_train_missing_subset, y_train_missing)
dtrain_1m_2m_subset =  xgb.DMatrix(x_train_1m_2m_subset)  # this is the 1m 2m loc
dtest_subset =  xgb.DMatrix(x_test_missing_subset)

cv_output = xgb.cv(xgb_params, dtrain_subset, 
                       num_boost_round=1000, 
                       early_stopping_rounds=20,
                       verbose_eval=50, show_stdv=False)
test_rmse = cv_output.loc[len(cv_output)-1,'test-rmse-mean']
print(test_rmse)

num_boost_rounds= len(cv_output)
print(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain_subset, num_boost_round= num_boost_rounds)
print('Finish XGBoost training')

y_train_1m_2m.shape

y_train_1m_2m = model.predict(dtrain_1m_2m_subset)
y_train_1m_2m_all_sq = (y_train_1m_2m)*x_train_1m_2m['full_sq']

y_train_1m_2m_all_sq.hist(bins=20)

train_copy.loc[(train_copy.price_doc==1000000)|(train_copy.price_doc==2000000),'price_doc'] = y_train_1m_2m_all_sq

import visualization as vis
vis.hist_density_plot(train_copy, x='price_doc', logx=True)

train_copy.to_csv('train_xgb_1m_2m_repredicted.csv')

train2 = train_cofpy

train2['price_full_sq'] = (train2['price_doc']/train2['full_sq'].astype(float)).astype(int)
y_train2 = train2['price_full_sq']

x_train2 = train2.drop(['price_full_sq','price_doc','id','timestamp'],axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0
}



x_train_missing_subset = x_train2
x_test_missing_subset  = x_test_missing

dtrain_subset = xgb.DMatrix(x_train_missing_subset, y_train2)
dtest_subset  = xgb.DMatrix(x_test_missing_subset)

cv_output = xgb.cv(xgb_params, dtrain_subset, 
                       num_boost_round=1000, 
                       early_stopping_rounds=20,
                       verbose_eval=50, show_stdv=False)
test_rmse = cv_output.loc[len(cv_output)-1,'test-rmse-mean']
print(test_rmse)

num_boost_rounds= len(cv_output)
print(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain_subset, num_boost_round= num_boost_rounds)
print('Finish XGBoost training')

y_predict = model.predict(dtest_subset)
y_predic_all_sq = (y_predict)*x_test_missing['full_sq']
df_sub = pd.DataFrame({'id': test_id, 'price_doc': y_predic_all_sq})
df_sub.to_csv('./sub_1.csv', index=False)
df_sub.head()

y_predict_mod = model.predict(dtest_subset)
y_predic_all_sq = (y_predict_mod)*x_test_missing['full_sq']*0.975
df_sub = pd.DataFrame({'id': test_id, 'price_doc': y_predic_all_sq})
df_sub.to_csv('./sub_2_mod.csv', index=False)
df_sub.head()

# 5437809/5685616.0

# sub_975=pd.read_csv('./sub_0.975_wei_0523.csv')

# r = sub_975.price_doc / y_predic_all_sq

# sub_975.head()

# r.hist(bins=20)

# sub_975.price_doc.sum() / y_predic_all_sq.sum()

# y_predict_mod_2 = model.predict(dtest_subset)
# y_predic_all_sq = (y_predict_mod_2)*x_test_missing['full_sq']*0.975*0.96
# df_sub = pd.DataFrame({'id': test_id, 'price_doc': y_predic_all_sq})
# df_sub.to_csv('./sub_3_mod.csv', index=False)
# df_sub.head()




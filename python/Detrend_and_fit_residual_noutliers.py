#Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from fbprophet import Prophet

from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge

from sklearn.metrics import mean_squared_error, make_scorer
from math import sqrt

from skopt import BayesSearchCV, gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt import dump

import lightgbm as lgb

from sklearn.pipeline import make_pipeline, make_union
#import category_encoders as en

from sklearn.ensemble import RandomForestRegressor

#Read test and train data - Please change this accordingly
train = pd.read_csv("../data/train_aWnotuB.csv", parse_dates=['DateTime'])
test = pd.read_csv("../data/test_BdBKkAj.csv",  parse_dates=['DateTime'])
sample_sub = pd.read_csv("../data/sample_submission_EZmX9uE.csv")

print(train.shape, test.shape, sample_sub.shape)

#Split data by junction values
def split_df(df):
    all_df = []
    for jn in df.Junction.unique():
        df_sub = df.loc[df.Junction == jn].set_index('DateTime')
        #print(jn)
        all_df.append(df_sub)
    return all_df

#Function to get features from date 
def get_datetime_feats(df):
    df['hourofday'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['day'] = df['DateTime'].dt.day
    df['dayofyear'] = df['DateTime'].dt.dayofyear
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    df['week'] = df['DateTime'].dt.week

#generate features related to date-time for all records in train and test
get_datetime_feats(train)
get_datetime_feats(test)

#Split data into individual time series
train1, train2, train3, train4 = split_df(train)
test1, test2, test3, test4 = split_df(test)

#Lets plot each of the timeseries
def plot_ts(df):
    #Whole series
    plt.figure(figsize=(12,12))
    plt.plot(df.Vehicles)
    plt.xticks(rotation=90)
    plt.show()
    
    #Last 4 weeks
    plt.figure(figsize=(12,12))
    plt.plot(df.Vehicles.iloc[-7*4*24:])
    plt.xticks(rotation=90)
    plt.show()
    
for df in [train1, train2, train3, train4]:
    plot_ts(df)

#Function Day and week counter (pseudo subsampling)
def get_dayweekcounter(train, test):
    train['day_counter'] = np.cumsum(np.abs(train['day'] - train['day'].shift()) > 0)
    test['day_counter'] = train['day_counter'].max() + np.cumsum(np.abs(test['day'] - test['day'].shift()) > 0)
    
    train['week_counter'] = np.cumsum(np.abs(train['week'] - train['week'].shift()) > 0)
    test['week_counter'] = train['week_counter'].max() + np.cumsum(np.abs(test['week'] - test['week'].shift()) > 0)

#Remove outliers - don't want one off things biasing our results
for df in [train1, train2, train3, train4]:
    df['Vehicles'] = df['Vehicles'].clip(df['Vehicles'].quantile(0.001), df['Vehicles'].quantile(0.999))

get_dayweekcounter(train1, test1)
lr1 = Ridge()
lr1.fit(train1[['day_counter']], train1['Vehicles'])
train1['trend'] = lr1.predict(train1[['day_counter']])
test1['trend'] = lr1.predict(test1[['day_counter']])

plt.plot(train1.day_counter, train1.Vehicles)
plt.plot(train1.day_counter, train1.trend)
plt.plot(test1.day_counter, test1.trend)
plt.show()

train1['residual'] = train1['Vehicles'] - train1['trend']
#test1['residual'] = test1['Vehicles'] - test1['trend']
plt.plot(train1.residual)
plt.xticks(rotation=90)
plt.show()

get_dayweekcounter(train2, test2)
lr2 = Ridge()

#We add square term as time series seem to follow a quadratic nature(WARNING: MIGHT OVERFIT)
#train2['day_counter_sqr'] = train2['day_counter']**2
#test2['day_counter_sqr'] = test2['day_counter']**2

lr2.fit(train2[['day_counter']], train2['Vehicles'])
train2['trend'] = lr2.predict(train2[['day_counter']])
test2['trend'] = lr2.predict(test2[['day_counter']])

plt.plot(train2.Vehicles)
plt.plot(train2.trend)
plt.plot(test2.trend)
plt.show()

train2['residual'] = train2['Vehicles'] - train2['trend']
#test1['residual'] = test1['Vehicles'] - test1['trend']
plt.plot(train2.residual)
plt.show()

get_dayweekcounter(train3, test3)
lr3 = HuberRegressor(epsilon=1.5) #We use Huber regressor because of lot of outliers
lr3.fit(train3[['day_counter']], train3['Vehicles'])
train3['trend'] = lr3.predict(train3[['day_counter']])
test3['trend'] = lr3.predict(test3[['day_counter']])

plt.plot(train3.Vehicles)
plt.plot(train3.trend)
plt.plot(test3.trend)
plt.show()

train3['residual'] = train3['Vehicles'] - train3['trend']
#test1['residual'] = test1['Vehicles'] - test1['trend']
plt.plot(train3.residual)
plt.show()

get_dayweekcounter(train4, test4)
lr4 = Ridge()
lr4.fit(train4[['day_counter']], train4['Vehicles'])
train4['trend'] = lr4.predict(train4[['day_counter']])
test4['trend'] = lr4.predict(test4[['day_counter']])

plt.plot(train4.Vehicles)
plt.plot(train4.trend)
plt.plot(test4.trend)
plt.show()

train4['residual'] = train4['Vehicles'] - train4['trend']
#test1['residual'] = test1['Vehicles'] - test1['trend']
plt.plot(train4.residual)
plt.show()

train_new = pd.concat([train1, train2, train3, train4]).reset_index()
test_new = pd.concat([test1, test2, test3, test4]).reset_index()
train_new.head()

#Get validation lists - March-June, Jan-June, May-June 

tr1, val1 = train_new.loc[train_new['DateTime'] < pd.to_datetime('2017/03/01')].index,             train_new.loc[train_new['DateTime'] >= pd.to_datetime('2017/03/01')].index
    
#tr2, val2 = train_new.loc[train_new['DateTime'] < pd.to_datetime('2017/01/01')].index, \
#            train_new.loc[train_new['DateTime'] >= pd.to_datetime('2017/01/01')].index
    
tr3, val3 = train_new.loc[train_new['DateTime'] < pd.to_datetime('2017/05/01')].index,             train_new.loc[train_new['DateTime'] >= pd.to_datetime('2017/05/01')].index
    
cvlist = [[tr1, val1], [tr3, val3]]

#Only pick useful features (check using cv)
feats = ['Junction', 'hourofday', 'dayofweek','day_counter', 'week_counter', 'day','dayofyear', 'week']
X = train_new[feats]
y = train_new['residual']
X_test = test_new[feats]

#Define sklearn compatible eval metric
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))
rmse_sk = make_scorer(rmse, False)

#Set up hyperparameter space for lightgbm optimization
def optimize_lightgbm(est, X, y, X_test, cvlist, weight_array, save_path="../utility/"):
    space  = [('regression', 'huber'),             # Objective
              (200, 1000),                           # n_estimators
              (7, 127),                              # num_leaves
              (0.4, 1),                            # colsample_bytree
              (0.4, 1),                            # subsample
              (20, 500),                             # min_child_samples
              (0, 1)]                               # alpha
    

    def objective(params):
        objective, n_estimators, num_leaves, colsample_bytree, subsample, min_child_samples, reg_alpha = params
        lgb_params = {
        'objective':objective,
        'n_estimators': n_estimators,
        'num_leaves': num_leaves,
        'colsample_bytree': colsample_bytree,
        'subsample': subsample,
        'min_child_samples': min_child_samples,
        'reg_alpha': reg_alpha 
        }

        print("parameters...................",params)
        scores = cross_val_score(est.set_params(**lgb_params), X, y, cv=cvlist, scoring=rmse_sk, verbose=1,
                                fit_params={'sample_weight':weight_array})
        print("scores........................",scores)
        #preds_test = est.set_params(**lgb_params).fit(X, y).predict_proba_corr(X_test)
        #preds_dict = {'params':lgb_params, 'train_preds': preds, 'test_preds':preds_test}
        #filepath = os.path.join(save_path, str(time.time()))
        #with open(filepath, "wb") as f:
        #    pickle.dump(preds_dict, f)
        #print("Saved..............:", filepath)
        return -1* np.mean(scores)

    res = gp_minimize(objective,                  # the function to minimize
                      space,                          # the bounds on each dimension of x
                      acq_func="EI",                  # the acquisition function
                      n_calls=30,                     # the number of evaluations of f 
                      n_random_starts=10,             # the number of random initialization points
                      random_state=1,
                     verbose=True) 
    
    return res

print("Performing hyperparameter optimization for lgb: Likely overfitting")
res_lgb = optimize_lightgbm(lgb.LGBMRegressor(learning_rate=0.01,
                                        max_depth = -1), X, y, X_test, cvlist, train_new['week_counter'])

#Fit on all data and predict on test
lgb_params = {
    'objective': res_lgb.x[0],
    'learning_rate': 0.05,
    'n_estimators': res_lgb.x[1],
    'num_leaves': res_lgb.x[2],
    'subsample': res_lgb.x[4],
    'colsample_bytree': res_lgb.x[3],
    'min_child_samples': res_lgb.x[5],
    'reg_alpha': res_lgb.x[6]
}
lgb1 = lgb.LGBMRegressor(**lgb_params)

lgb1.fit(X, y)
lgb_preds = lgb1.predict(X_test)

test_new['Vehicles'] = test_new['trend'] + lgb_preds
sns.distplot(test_new['Vehicles'])
plt.show()

#plot timeseries
def plot_ts(train, test):
    #Whole series
    plt.figure(figsize=(14,8))
    plt.plot(test.set_index('DateTime').Vehicles)

    plt.plot(train.set_index('DateTime').Vehicles.iloc[-7*8*24:])
    plt.xticks(rotation=90)
    plt.show()
    
for jn in [1,2,3,4]:
    plot_ts(train_new.loc[train_new.Junction == jn], test_new.loc[test_new.Junction == jn])

submission_df = test_new[['ID', 'Vehicles']]
submission_df.head()

submission_df.to_csv("../utility/detrend_lgb_v1.csv", index=False)




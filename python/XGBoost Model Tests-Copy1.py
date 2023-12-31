import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import scipy as sp
import os
import xgboost as xgb
get_ipython().magic('matplotlib inline')

train = pd.read_csv("./train.csv", parse_dates = ['timestamp'])
print train.shape
test = pd.read_csv("./test.csv", parse_dates = ['timestamp'])
print test.shape
macro = pd.read_csv("./macro.csv", parse_dates = ['timestamp'])
print macro.shape
train.head()

# Merge the train and test with macro
train_full_set = pd.merge(train, macro, how = 'left', on = 'timestamp')
test_full_set = pd.merge(test, macro, how = 'left', on = 'timestamp')

pd.set_option('display.max_columns', None)

train_full_set.head()

#train_full_set.apply(lambda x: type(x[0]))
type(train_full_set)

# Let's do property features again, but this time with all macros
def features_selection(df):
    '''
    Select the features that we want to include from original training/test/macro set
    '''
    features = ['timestamp', 'full_sq', 'life_sq', 'floor', 
                        'max_floor', 'material', 'build_year', 'num_room',
                        'kitch_sq', 'state', 'price_doc']
    df = df[features]
    return df

train_housing_features = features_selection(train_full_set)
test_features = ['timestamp', 'full_sq', 'life_sq', 'floor', 
                        'max_floor', 'material', 'build_year', 'num_room',
                        'kitch_sq', 'state']
#train_housing_features = features_selection(train)
test_housing_features = test_full_set[test_features]

# with just the training set and test set (not merged with macro)
train = features_selection(train)
test = test[test_features]

train_housing_features.head()

test_housing_features.head()

# what's missing?
print np.sum(train_housing_features.isnull())
print train_housing_features.shape

# Let's merge the entire macro set with the train/test feature sets
train_df = pd.merge(train_housing_features, macro, how = 'left', on = 'timestamp')
test_df = pd.merge(test_housing_features, macro, how = 'left', on = 'timestamp')

train_df.head()

## Do some encoding to any categorical variables
from sklearn.preprocessing import LabelEncoder

def encode_object_features(train, test):
    '''(DataFrame, DataFrame) -> DataFrame, DataFrame
    
    Will encode each non-numerical column.
    '''
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    cols_to_encode = train.select_dtypes(include=['object'], exclude=['int64', 'float64']).columns
    for col in cols_to_encode:
        le = LabelEncoder()
        #Fit encoder
        le.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        #Transform
        train[col] = le.transform(list(train[col].values.astype('str')))
        test[col] = le.transform(list(test[col].values.astype('str')))
    
    return train, test

train_df, test_df = encode_object_features(train_df, test_df)

train, test = encode_object_features(train, test)

test.head()

# Do stuff with the date
def add_date_features(df):
    '''(DataFrame) -> DataFrame
    
    Will add some specific columns based on the date
    of the sale.
    '''
    #Convert to datetime to make extraction easier
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    #Extract features
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['year'] = df['timestamp'].dt.year
    
    #These features inspired by Bruno's Notebook at https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
    #Month-Year
    #month_year = df['timestamp'].dt.month + df['timestamp'].dt.year * 100
    #month_year_map = month_year.value_counts().to_dict()
    #df['month_year'] = month_year.map(month_year_map)
    #Week-Year
    week_year = df['timestamp'].dt.weekofyear + df['timestamp'].dt.year * 100
    week_year_map = week_year.value_counts().to_dict()
    df['week_year'] = week_year.map(week_year_map)
    df.drop('timestamp', axis=1, inplace=True)
    return df

train_df = add_date_features(train_df)
test_df = add_date_features(test_df)

train = add_date_features(train)
test = add_date_features(test)

train_df.head()

train.head()

#Get Data
Y_train = np.log(train_df['price_doc']).values
X_train = train_df.ix[:, train_df.columns != 'price_doc'].values
X_test = test_df.values

print(Y_train.shape)
print(X_train.shape)
print(X_test.shape)

#Get Data
y_train = np.log(train['price_doc']).values
x_train = train_df.ix[:, train.columns != 'price_doc'].values
x_test = test.values

print(y_train.shape)
print(x_train.shape)
print(x_test.shape)

# Do what Shu did. Create a cross validation set manually

# Create a validation set, with last 20% of data
size_ = 7000
X_train_sub, Y_train_sub = X_train[:-size_],  Y_train[:-size_]
X_val, Y_val = X_train[-size_:],  Y_train[-size_:]


dtrain = xgb.DMatrix(X_train, 
                    Y_train)
dtrain_sub = xgb.DMatrix(X_train_sub, 
                        Y_train_sub)
d_val = xgb.DMatrix(X_val, 
                    Y_val)
dtest = xgb.DMatrix(X_test)

# Create a validation set, with last 20% of data
size_ = 7000
x_train_sub, y_train_sub = x_train[:-size_],  y_train[:-size_]
x_val, y_val = x_train[-size_:],  y_train[-size_:]


dtrain = xgb.DMatrix(x_train, 
                    y_train)
dtrain_sub = xgb.DMatrix(x_train_sub, 
                        y_train_sub)
d_val = xgb.DMatrix(x_val, 
                    y_val)
dtest = xgb.DMatrix(x_test)

# hyperparameters
xgb_params = {
    'eta': 0.02,
    'max_depth': 5,
    'subsample': .8,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

sub_model = xgb.train(xgb_params, 
                      dtrain_sub, 
                      num_boost_round=2000,
                      evals=[(d_val, 'val')],
                      early_stopping_rounds=20, 
                      verbose_eval=20)

# hyperparameters
xgb_params = {
    'eta': 0.02,
    'max_depth': 5,
    'subsample': .8,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

sub_model = xgb.train(xgb_params, 
                      dtrain_sub, 
                      num_boost_round=2000,
                      evals=[(d_val, 'val')],
                      early_stopping_rounds=20, 
                      verbose_eval=20)

# Check importance
fig, ax = plt.subplots(1, 1, figsize=(8, 16))
xgb.plot_importance(sub_model, ax=ax)

# Check importance
fig, ax = plt.subplots(1, 1, figsize=(8, 16))
xgb.plot_importance(sub_model, ax=ax)

#from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# Try cross-validating the other way.
#Initialize Model
xgb_reg = XGBRegressor()
#Create cross-validation
cv = TimeSeriesSplit(n_splits=5)
#Train & Test Model
cross_val_results = cross_val_score(xgb_reg, X_train, Y_train, cv=cv, scoring='neg_mean_squared_error')
print(cross_val_results.mean())




XGBRegressor().get_params().keys()

np.arange(0,1,.1)




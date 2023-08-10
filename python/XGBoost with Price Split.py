import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.preprocessing import scale

get_ipython().magic('matplotlib inline')

df_train = pd.read_csv("../Sberbank/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../Sberbank/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../Sberbank/macro.csv", parse_dates=['timestamp'])
state_build = pd.read_csv('../EDA/merged_w_state_build_2017-05-30.csv')
df_train.head()

# =============================
# =============================
# cleanup
# brings error down a lot by removing extreme price per sqm
print(df_train.shape)
# df_train.loc[df_train.full_sq == 0, 'full_sq'] = 30
print(df_train.shape)
# =============================
# =============================

frames = [df_train, df_test]

df = pd.concat(frames)

df['full_sq^2'] = state_build['full_sq^2']
df['age'] = state_build['age']
df['state'] = state_build['state']

df['age_log'] = np.log1p(df['age'])

# Add month-year
month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df['month'] = df.timestamp.dt.month
df['dow'] = df.timestamp.dt.dayofweek

# Other feature engineering
df['rel_floor'] = df['floor'] / df['max_floor'].astype(float)
df['rel_kitch_sq'] = df['kitch_sq'] / df['full_sq'].astype(float)


# Feature engineering
df['kitch_life'] = df.kitch_sq/df.life_sq
df['extra_on_life'] = df.full_sq/df.life_sq
df['rel_floor'] = df.floor/df.max_floor



# Separate dtypes
df_numeric = df.select_dtypes(exclude=['object', 'datetime'])    
df_obj = df.select_dtypes(include=['object', 'datetime']).copy()


# Deal with categorical values
for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]    # how is this different from above????


# Log transform skewed numeric features 
get_col = df_numeric.dtypes[(df_numeric.dtypes == "int64") | (df_numeric.dtypes == "float64")].index
get_skews = df_numeric[get_col].apply(lambda x: skew(x.dropna()))
get_skews = get_skews[get_skews>0.5]
get_skews = get_skews.index
df_numeric[get_skews] = np.log1p(df_numeric[get_skews])       


# concatenate back    
df_values = pd.concat([df_numeric, df_obj], axis=1)

# features to not use

not_important = ['id', 'timestamp', 'full_sq','max_floor', 'build_year']

for feature in not_important:
    df = df.loc[:, df.columns != feature]

from sklearn import preprocessing

for feature in df_values.columns:
    if df_values[feature].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_values[feature].values)) 
        df_values[feature] = lbl.transform(list(df_values[feature].values))

# df = df_values.dropna()

# impute missing values with mean
df_values = df_values.loc[:, df_values.columns != 'price_doc'].apply(lambda x: x.fillna(x.median()),axis=0) # newdf is the numeric columns

for feature in df_values.columns:
    if np.sum(df_values[feature].isnull()) > 0:
        print feature,np.sum(df_values[feature].isnull())

# add back price_doc

df_values['price_doc'] = df['price_doc']

df_train = df_values.loc[df_values['price_doc'].notnull()]
df_test = df_values.loc[df_values['price_doc'].isnull()]

df_train.shape, df_test.shape

y_train = df_train.loc[df_train['price_doc'].notnull(), 'price_doc']
X_train = df_train.loc[df_train['price_doc'].notnull(), df_train.columns != 'price_doc']
X_test = df_test.loc[df_test['price_doc'].isnull(), df_test.columns != 'price_doc']

X_train.shape, y_train.shape, X_test.shape

# Save the column names for features names
df_columns = X_train.columns


# Set the parameters
xgb_params = {
    'eta': 0.03,   # 0.05 orig
    'max_depth': 2,    # 5 orig
    'subsample': 0.7,     # 0.7 orig
    'colsample_bytree': 0.7,     # 0.7 orig
    'objective': 'reg:linear',    
    'eval_metric': 'rmse',
    'silent': 1
}

# Train the set against the actual prices and then mak predictions
dtrain = xgb.DMatrix(X_train.values, y_train.values, feature_names=df_columns)
dtest = xgb.DMatrix(X_test.values, feature_names=df_columns)

X_train.columns

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=50,
   verbose_eval=True, show_stdv=False)
cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
num_boost_rounds = len(cv_result)

num_boost_rounds = 600

# Tune XGB `num_boost_rounds`

# Run the model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# Plot the feature importance
fig, ax = plt.subplots(1, 1, figsize=(8, 54))
xgb.plot_importance(model, height=0.5, ax=ax)


# Make the predictions
y_pred = model.predict(dtest)


# Save the csv
# df_sub.to_csv('sub_param_tuned_3_pradeep.csv', index=False)

y_pred

split_value = 11000000 # determined by normality 

df_train_high = df_train.loc[df_train['price_doc'] > split_value, :]
df_test_high = df_test.loc[df_train['price_doc'] > split_value, :]

y_train_high = df_train_high['price_doc']

df_train_low = df_train.loc[df_train['price_doc'] < split_value, :]
df_test_low = df_test.loc[df_train['price_doc'] < split_value, :]

y_train_low = df_train_low['price_doc']

y_train_high = df_train_high.loc[df_train_high['price_doc'].notnull(), 'price_doc']
X_train_high = df_train_high.loc[df_train_high['price_doc'].notnull(), df_train_high.columns != 'price_doc']
X_test_high = df_test_high.loc[df_test_high['price_doc'].isnull(), df_test_high.columns != 'price_doc']

y_train_high = np.log1p(y_train_high)

# Save the column names for features names
df_columns = X_train_high.columns


# Set the parameters
xgb_params = {
    'eta': 0.03,   # 0.05 orig
    'max_depth': 2,    # 5 orig
    'subsample': 0.7,     # 0.7 orig
    'colsample_bytree': 0.7,     # 0.7 orig
    'objective': 'reg:linear',    
    'eval_metric': 'rmse',
    'silent': 1
}

# Train the set against the actual prices and then make predictions
dtrain = xgb.DMatrix(X_train_high.values, y_train_high.values, feature_names=df_columns)
dtest = xgb.DMatrix(X_test_high.values, feature_names=df_columns)

X_train_high.shape, y_train.shape

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=50,
   verbose_eval=True, show_stdv=False)
cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
num_boost_rounds = len(cv_result)

num_boost_rounds = 800

# Tune XGB `num_boost_rounds`

# Run the model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# Plot the feature importance
fig, ax = plt.subplots(1, 1, figsize=(8, 54))
xgb.plot_importance(model, height=0.5, ax=ax)


# Make the predictions
y_pred = model.predict(dtest)


# Save the csv
# df_sub.to_csv('sub_param_tuned_3_pradeep.csv', index=False)

y_test_high = y_pred



y_train_low = df_train_low.loc[df_train_low['price_doc'].notnull(), 'price_doc']
X_train_low = df_train_low.loc[df_train_low['price_doc'].notnull(), df_train_low.columns != 'price_doc']
X_test_low = df_test_low.loc[df_test_low['price_doc'].isnull(), df_test_low.columns != 'price_doc']

# Save the column names for features names
df_columns = X_train_low.columns


# Set the parameters
xgb_params = {
    'eta': 0.03,   # 0.05 orig
    'max_depth': 2,    # 5 orig
    'subsample': 0.7,     # 0.7 orig
    'colsample_bytree': 0.7,     # 0.7 orig
    'objective': 'reg:linear',    
    'eval_metric': 'rmse',
    'silent': 1
}

# Train the set against the actual prices and then make predictions
dtrain = xgb.DMatrix(X_train_low.values, y_train_low.values, feature_names=df_columns)
dtest = xgb.DMatrix(X_test_low.values, feature_names=df_columns)

X_train_low.shape, y_train.shape

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=50,
   verbose_eval=True, show_stdv=False)
cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
num_boost_rounds = len(cv_result)

num_boost_rounds = 500

# Tune XGB `num_boost_rounds`

# Run the model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# Plot the feature importance
fig, ax = plt.subplots(1, 1, figsize=(8, 54))
xgb.plot_importance(model, height=0.5, ax=ax)


# Make the predictions
y_pred = model.predict(dtest)


# Save the csv
# df_sub.to_csv('sub_param_tuned_3_pradeep.csv', index=False)

y_test_low = y_pred

len(y_test_high), len(y_test_low),  len(y_test_high) + len(y_test_low)

len(X_test_high), len(X_test_low),  len(X_test_high) + len(X_test_low)

X_test_high['price_doc'] = np.expm1(y_test_high)
X_test_low['price_doc'] = y_test_low

frames = [X_test_low, X_test_high]

X_test = pd.concat(frames)

frames = [X_train_low, X_train_high]

X_train = pd.concat(frames)

frames = [y_train_low, y_train_high]

y_train = pd.concat(frames)

y_pred = X_test[['id','price_doc']]

sample = pd.read_csv('../EDA/submissions/31310.csv')

sample = sample.merge(y_pred, left_on='id',right_on='id',how='left')

sample.loc[sample['price_doc_y'].isnull(), 'price_doc_y'] = sample.loc[sample['price_doc_y'].isnull(), 'price_doc_x']

# sample['price_doc'] = (sample['price_doc_x'] + sample['price_doc_y'])/2

sample

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

get_ipython().magic('matplotlib inline')


plt.figure(figsize=(12,4),dpi=200)
sns.kdeplot(sample['price_doc_x'].rolling(window=300).mean(), label='.31310',lw=3)
sns.kdeplot(sample['price_doc_y'].rolling(window=300).mean(), label='new',lw=3)
# sns.kdeplot(sample['price_doc'].rolling(window=300).mean(), label='mean',lw=3)
plt.xlabel('Predicted Price Distribution')
plt.ylabel('Density')
plt.legend(title='Score')
plt.show()

plt.figure(figsize=(25,10),dpi=200)
plt.plot(sample['price_doc_x'].rolling(window=700).mean(), label='.31310', alpha =0.6, lw=1)
plt.plot(sample['price_doc_y'].rolling(window=700).mean(), label='New', alpha =0.6, lw=1)
# plt.plot(sample['price_doc'].rolling(window=700).mean(), label='Mean', alpha =1, lw=2)
plt.xlabel('Distribution')
plt.ylabel('Predicted Price')
plt.legend(title='Score')
plt.show()

submission = sample[['id', 'price_doc']]
submission.set_index('id')

submission['price_doc'] = submission['price_doc']*.97

submission.to_csv('../EDA/submissions/submission_053117.csv')

submission = submission.set_index('id')

submit = pd.read_csv('../EDA/submissions/weighted_final_053017.csv', index_col='id')

submission

submit['price_doc'] = sample['price_doc']

submission.to_csv('../EDA/submissions/submission_053117.csv', index=True)




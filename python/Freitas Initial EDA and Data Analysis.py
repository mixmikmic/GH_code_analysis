import pandas as pd
import numpy as np
import scipy as sp
import sklearn

#import data and merge the macro onto the train and test
train_df = pd.read_csv("train.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("test.csv", parse_dates=['timestamp'])
macro_df = pd.read_csv("macro.csv", parse_dates=['timestamp'])
train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')
print(train_df.shape, test_df.shape)

#truncate the extreme values in price_doc
ulimit = np.percentile(train_df.price_doc.values, 99)
llimit = np.percentile(train_df.price_doc.values, 1)
train_df['price_doc'].loc[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].loc[train_df['price_doc']<llimit] = llimit

#import data and merge the macro onto the train and test
trainsm_df = pd.read_csv("train_small.csv", parse_dates=['timestamp'])
testsm_df = pd.read_csv("test_small.csv", parse_dates=['timestamp'])
macrosm_df = pd.read_csv("macro_small.csv", parse_dates=['timestamp'])
trainsm_df = pd.merge(trainsm_df, macrosm_df, how='left', on='timestamp')
testsm_df = pd.merge(testsm_df, macrosm_df, how='left', on='timestamp')
print(trainsm_df.shape, testsm_df.shape)

#truncate the extreme values in price_doc
ulimit = np.percentile(trainsm_df.price_doc.values, 99)
llimit = np.percentile(trainsm_df.price_doc.values, 1)
trainsm_df['price_doc'].loc[trainsm_df['price_doc']>ulimit] = ulimit
trainsm_df['price_doc'].loc[trainsm_df['price_doc']<llimit] = llimit

okurg_df = pd.read_csv("okurg.csv")

trainsm_df = pd.merge(trainsm_df, okurg_df, how='left', on='sub_area')
testsm_df = pd.merge(testsm_df, okurg_df, how='left', on='sub_area')

trainsm_df.head()

dtype_df = trainsm_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

from sklearn import model_selection, preprocessing

for f in trainsm_df.columns:
    if trainsm_df[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(trainsm_df[f].values.astype('str')) + list(testsm_df[f].values.astype('str')))
        trainsm_df[f] = lbl.transform(list(trainsm_df[f].values.astype('str')))
        testsm_df[f] = lbl.transform(list(testsm_df[f].values.astype('str')))

trainsm_df["gender_ratio"] = trainsm_df["female_f"] / trainsm_df["full_all"].astype("float")
testsm_df["gender_ratio"] = testsm_df["female_f"] / testsm_df["full_all"].astype("float")
testsm_df.head()

testsm_df.head()
trainsm_df.head()
#looks like one of the ratios for elder/youth is more than one... would need to fix

trainsm_df["gender_ratio_work"] = trainsm_df["work_female"] / trainsm_df["work_all"].astype("float")
testsm_df["gender_ratio_work"] = testsm_df["work_female"] / testsm_df["work_all"].astype("float")

trainsm_df["elder_ratio"] = trainsm_df["ekder_all"] / trainsm_df["full_all"].astype("float")
testsm_df["elder_ratio"] = testsm_df["ekder_all"] / testsm_df["full_all"].astype("float")

trainsm_df["elder_ratio"] = trainsm_df["ekder_all"] / (trainsm_df["young_all"] + trainsm_df["work_all"] + trainsm_df["ekder_all"]).astype("float")
testsm_df["elder_ratio"] = testsm_df["ekder_all"] / testsm_df["full_all"].astype("float")

trainsm_df["youth_ratio"] = trainsm_df["young_all"] / (trainsm_df["young_all"] + trainsm_df["work_all"] + trainsm_df["ekder_all"]).astype("float")
testsm_df["youth_ratio"] = testsm_df["young_all"] / testsm_df["full_all"].astype("float")

trainsm_df["pop_density"] = trainsm_df["raion_popul"] / trainsm_df["area_m"].astype("float")
testsm_df["pop_density"] = testsm_df["raion_popul"] / testsm_df["area_m"].astype("float")



print(trainsm_df["material"].value_counts(dropna=False))
print(testsm_df["material"].value_counts(dropna=False))

trainsm_df.loc[trainsm_df["material"] == 3, "material"] = np.nan
testsm_df.loc[testsm_df["material"] == 3, "material"] = np.nan

print(trainsm_df['state'].value_counts(dropna=False))
print(testsm_df['state'].value_counts(dropna=False))

trainsm_df.loc[trainsm_df["state"] == 33, "state"] = 3

trainsm_df.loc[trainsm_df['max_floor'] == 117, "max_floor"] = 17
trainsm_df.loc[trainsm_df['max_floor'] > 60, "max_floor"] = np.nan
#putting the suspicious ones into NaNs

trainsm_df.loc[trainsm_df['floor'] > 60]

trainsm_df.loc[trainsm_df['floor'] == 77, "floor"] = 7
#fixing the floor variable

trainsm_df['build_year'].value_counts(dropna=False).sort_index()
testsm_df['build_year'].value_counts(dropna=False).sort_index()

trainsm_df.loc[trainsm_df['build_year'] == 20052009, 'build_year'] = 2005
trainsm_df.loc[trainsm_df['build_year'] == 215, 'build_year'] = 2015
trainsm_df.loc[trainsm_df['build_year'] == 4965, 'build_year'] = 1965
trainsm_df.loc[trainsm_df['build_year'] == 71, 'build_year'] = 1971
trainsm_df.loc[trainsm_df['build_year'] < 1800, 'build_year'] = np.nan

testsm_df.loc[testsm_df['build_year'] == 215, 'build_year'] = 2015
testsm_df.loc[testsm_df['build_year'] < 1800, 'build_year'] = np.nan



## num_room
trainsm_df.loc[trainsm_df['num_room'] > 9]
#converting the number of rooms (like 10) to missing values
#you can see that the life sq is the same as the num room

trainsm_df.loc[trainsm_df['num_room'] > 9, 'num_room'] = np.nan

## num_room
testsm_df.loc[testsm_df['num_room'] > 9, 'num_room'] = np.nan

## full_sq 
trainsm_df.loc[trainsm_df['full_sq'] > 300]

#how can we make the full sq better?

trainsm_df.loc[trainsm_df['full_sq'] > 1000, 'full_sq'] = trainsm_df.loc[trainsm_df['full_sq'] > 1000, 'full_sq']/100
trainsm_df.loc[trainsm_df['full_sq'] > 310, 'full_sq'] = trainsm_df.loc[trainsm_df['full_sq'] > 310, 'full_sq']/10

#Imputation to make the square footage make sense
#keeping the houses that have a realistic number

## full_sq test
testsm_df.loc[testsm_df['full_sq'] > 400, 'full_sq'] = testsm_df.loc[testsm_df['full_sq'] > 400, 'full_sq'] / 10

trainsm_df.loc[(trainsm_df['full_sq'] < trainsm_df['life_sq']) & (trainsm_df['life_sq'] > 100)]

#when full square is bigger than life sq
#when life square is huge

## life_sq
trainsm_df.loc[13549, 'life_sq'] = trainsm_df.loc[13549, 'life_sq'] / 100

rows = (trainsm_df['full_sq'] < trainsm_df['life_sq']) & (trainsm_df['life_sq'] > 100)
trainsm_df.loc[rows, 'life_sq'] = trainsm_df.loc[rows, 'life_sq'] / 10

#make sure that the life square makes sense

## life_sq test
rows2 = (testsm_df['full_sq'] < testsm_df['life_sq']) & (testsm_df['life_sq'] > 110)
testsm_df.loc[rows2]

testsm_df.loc[rows2, 'life_sq'] = testsm_df.loc[rows2, 'life_sq'] / 10

#also had to fix the test data--impute the data

#kitchen:
rows3 = (trainsm_df['kitch_sq'] > trainsm_df['full_sq']) & (trainsm_df['kitch_sq'] > 100)
trainsm_df.loc[rows3]

trainsm_df.loc[13120, 'build_year'] = 1970
trainsm_df.loc[11523, 'kitch_sq'] = trainsm_df.loc[11523, 'kitch_sq'] / 100
rows8 = (trainsm_df['kitch_sq'] > trainsm_df['full_sq']) & (trainsm_df['kitch_sq'] > 100)
trainsm_df.loc[rows8, 'kitch_sq'] = np.nan

#can use one of the eyars to impute the build year, get rid of some, and fix others

rows4 = (testsm_df['kitch_sq'] > testsm_df['full_sq']) & (testsm_df['kitch_sq'] > 100)
testsm_df.loc[rows4]

testsm_df.loc[rows4, 'kitch_sq'] = np.nan

import matplotlib.pyplot as plt
import matplotlib
from pandas.plotting import scatter_matrix
get_ipython().magic('matplotlib inline')

#making a new property called sq meter, taking the maximum value of the full or life (because of na's)

## property square meters
trainsm_df.loc[:, 'sq_metr'] = trainsm_df.loc[:, ['full_sq','life_sq']].max(axis=1)
trainsm_df.loc[trainsm_df['sq_metr'] < 6, 'sq_metr'] = np.nan
#make sure that the zeros are gone
trainsm_df.plot.scatter(x='sq_metr', y='price_doc')

## remove outlier (sq_metr)
trainsm_df = trainsm_df.drop(trainsm_df[trainsm_df['sq_metr'] > 600].index)
#get rid of the one outlier
trainsm_df.plot.scatter(x='sq_metr', y='price_doc')

## property square meters for Test Data
testsm_df.loc[:, 'sq_metr'] = testsm_df.loc[:, ['full_sq','life_sq']].max(axis=1)
testsm_df.loc[testsm_df['sq_metr'] < 6, 'sq_metr'] = np.nan

#check to see if there are any crazy outliers in the test data (there are not)
trainsm_df['sq_metr'].value_counts(dropna=False).sort_index()
testsm_df['sq_metr'].value_counts(dropna=False).sort_index()



## adding a log transformation column to the training set:
trainsm_df.loc[:, 'log_price_doc'] = np.log(trainsm_df['price_doc'] + 1)
#make sure that the data is positive -- the log will give a very negative value if it is close to one
trainsm_df.head()











#import xgboost as xgb

#This is to use xgboost once you figure out how to install it


for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


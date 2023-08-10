import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import scipy as sp
import os
import xgboost as xgb
get_ipython().magic('matplotlib inline')

pd.set_option('display.max_columns', None)

# load dataset
train = pd.read_csv("../Sberbank/train.csv", parse_dates=['timestamp'], index_col=False)
test = pd.read_csv("../Sberbank/test.csv", parse_dates=['timestamp'], index_col=False)
macro = pd.read_csv("../Sberbank/macro.csv", parse_dates=['timestamp'], index_col=False)

train.head()

np.sum(train.isnull())

ftr_train = train.loc[:, ['price_doc',
'timestamp',
'full_sq',
'life_sq',
'floor',
'max_floor',
'material',
'build_year',
'num_room',
'kitch_sq',
'state',
'product_type',
'sub_area',
'indust_part',
'school_education_centers_raion',
'sport_objects_raion',
'culture_objects_top_25_raion',
'oil_chemistry_raion',
'metro_min_avto',
'green_zone_km',
'industrial_km',
'kremlin_km',
'radiation_km',
'ts_km',
'fitness_km',
'stadium_km',
'additional_education_km',
'cafe_count_1500_price_500',
'cafe_count_1500_price_high',
'cafe_count_2000_price_2500',
'trc_sqm_5000',
'cafe_count_5000',
'cafe_count_5000_price_high']]


ftr_test = test.loc[:, ['timestamp',
'full_sq',
'life_sq',
'floor',
'max_floor',
'material',
'build_year',
'num_room',
'kitch_sq',
'state',
'product_type',
'sub_area',
'indust_part',
'school_education_centers_raion',
'sport_objects_raion',
'culture_objects_top_25_raion',
'oil_chemistry_raion',
'metro_min_avto',
'green_zone_km',
'industrial_km',
'kremlin_km',
'radiation_km',
'ts_km',
'fitness_km',
'stadium_km',
'additional_education_km',
'cafe_count_1500_price_500',
'cafe_count_1500_price_high',
'cafe_count_2000_price_2500',
'trc_sqm_5000',
'cafe_count_5000',
'cafe_count_5000_price_high']]

ftr_macro = macro.loc[:,['timestamp',
'oil_urals',
'gdp_quart',
'cpi',
'ppi',
'usdrub',
'eurrub',
'gdp_annual',
'rts',
'micex',
'micex_cbi_tr',
'deposits_rate',
'mortgage_rate',
'income_per_cap',
'salary',
'labor_force',
'unemployment',
'employment']]

train = pd.merge(ftr_train, ftr_macro, how = 'left', on = 'timestamp')
test = pd.merge(ftr_test, ftr_macro, how = 'left', on = 'timestamp')

print train.shape
print test.shape

#train.to_csv('sberbank_train_unclean.csv')

#test.to_csv('sberbank_test_unclean.csv')

train.apply(lambda x: type(x[0]))

test.apply(lambda x: type(x[0]))

np.sum(train.isnull())

from sklearn import model_selection, preprocessing
from sklearn.preprocessing import LabelEncoder

def encode_object_features(train_df, test_df):
    '''(DataFrame, DataFrame) -> DataFrame, DataFrame
    
    Will encode each non-numerical column.
    '''
    for f in train_df.columns:
        if train_df[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))
            train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))
            test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))
    
    return train_df, test_df

train, test = encode_object_features(train, test)

# Impute missing values in training set

from scipy.stats import mode

print int(mode(train['life_sq']).mode[0])
print mode(train['floor']).mode[0]
print mode(train['max_floor']).mode[0]
print mode(train['material']).mode[0]
print mode(train['build_year']).mode[0]
print mode(train['num_room']).mode[0]
print mode(train['kitch_sq']).mode[0]
print mode(train['state']).mode[0]
print mode(train['industrial_km']).mode[0]


train['life_sq'].fillna(mode(train['life_sq']).mode[0], inplace=True)
train['floor'].fillna(mode(train['floor']).mode[0], inplace=True)
train['max_floor'].fillna(mode(train['max_floor']).mode[0], inplace=True)
train['material'].fillna(mode(train['material']).mode[0], inplace=True)
train['build_year'].fillna(mode(train['build_year']).mode[0], inplace=True)
train['num_room'].fillna(mode(train['num_room']).mode[0], inplace=True)
train['kitch_sq'].fillna(mode(train['kitch_sq']).mode[0], inplace=True)
train['state'].fillna(mode(train['state']).mode[0], inplace=True)
train['industrial_km'].fillna(mode(train['industrial_km']).mode[0], inplace=True)

print test.shape
np.sum(test.isnull())

# Impute missing values in test set
print mode(test['life_sq']).mode[0]
print mode(test['build_year']).mode[0]
print mode(test['state']).mode[0]
print mode(test['income_per_cap']).mode[0]
print mode(test['salary']).mode[0]
print mode(test['labor_force']).mode[0]
print mode(test['unemployment']).mode[0]
print mode(test['employment']).mode[0]


test['life_sq'].fillna(mode(test['life_sq']).mode[0], inplace=True)
test['build_year'].fillna(mode(test['build_year']).mode[0], inplace=True)
test['state'].fillna(mode(test['state']).mode[0], inplace=True)
test['income_per_cap'].fillna(mode(test['income_per_cap']).mode[0], inplace=True)
test['salary'].fillna(mode(test['salary']).mode[0], inplace=True)
test['labor_force'].fillna(mode(test['labor_force']).mode[0], inplace=True)
test['unemployment'].fillna(mode(test['unemployment']).mode[0], inplace=True)
test['employment'].fillna(mode(test['employment']).mode[0], inplace=True)

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
    df['year_month'] = df['timestamp'].apply(lambda x:x.strftime('%Y%m'))
    #Week-Year
    #df['week_year'] = df['timestamp'].dt.weekofyear
    #df.drop('timestamp', axis=1, inplace=True)
    return df

train = add_date_features(train)
test = add_date_features(test)

np.log(train[[0]])

train.shape

test.head()

# I wanna add some lag features to take into account the time series dependencies in the model
price_train = train.groupby('year_month')[['price_doc']].mean()
plt.plot(price_train.index, price_train['price_doc'], color = 'r')

# I wanna add some lag features to take into account the time series dependencies in the model
eurrub_train = train.groupby('year_month')[['eurrub']].mean()
eurrub_test = test.groupby('year_month')[['eurrub']].mean()

plt.plot(eurrub_train.index, eurrub_train['eurrub'], color = 'r')
plt.plot(eurrub_test.index, eurrub_test['eurrub'], color = 'b')

usdrub_train = train.groupby('year_month')[['usdrub']].mean()
usdrub_test = test.groupby('year_month')[['usdrub']].mean()

plt.plot(usdrub_train.index, usdrub_train['usdrub'], color = 'r')
plt.plot(usdrub_test.index, usdrub_test['usdrub'], color = 'b')

micex_train = train.groupby('year_month')[['micex_cbi_tr']].mean()
micex_test = test.groupby('year_month')[['micex_cbi_tr']].mean()

plt.plot(micex_train.index, micex_train['micex_cbi_tr'], color = 'r')
plt.plot(micex_test.index, micex_test['micex_cbi_tr'], color = 'b')

oil_train = train.groupby('year_month')[['oil_urals']].mean()
oil_test = test.groupby('year_month')[['oil_urals']].mean()

plt.plot(oil_train.index, oil_train['oil_urals'], color = 'r')
plt.plot(oil_test.index, oil_test['oil_urals'], color = 'b')

inflation_train = train.groupby('year_month')[['cpi']].mean()
inflation_test = test.groupby('year_month')[['cpi']].mean()

plt.plot(inflation_train.index, inflation_train['cpi'], color = 'r')
plt.plot(inflation_test.index, inflation_test['cpi'], color = 'b')

labor_train = train.groupby('year_month')[['labor_force']].mean()
labor_test = test.groupby('year_month')[['labor_force']].mean()

plt.plot(labor_train.index, labor_train['labor_force'], color = 'r')
plt.plot(labor_test.index, labor_test['labor_force'], color = 'b')

# Oil Urals Lag
from pandas import Series
x = train.groupby('year_month')['oil_urals'].mean()
# type(x)
differenced = x.shift(12)
# trim off the first year of empty data
#differenced = differenced[12:]
# plot differenced dataset
#print x.head(20)
#print differenced.head(20)
differenced.plot()

differenced.shift(12)

# Oil Urals Lag
from pandas import Series
x_1 = train.groupby('year_month')['oil_urals'].mean()
# type(x)
differenced = x_1.diff(1)
# trim off the first year of empty data
differenced = differenced[1:]
# plot differenced dataset
print type(x_1)
differenced.plot()


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(differenced)
pyplot.show()

# usdrub Urals Lag
from pandas import Series
rub_1 = train.groupby('year_month')['usdrub'].mean()
# type(x)
differenced_1 = rub_1.diff(1)
# trim off the first year of empty data
differenced_1 = differenced_1[1:]
# plot differenced dataset
differenced_1.plot()

plot_acf(differenced_1)
pyplot.show()

labor_force_1 = train.groupby('year_month')['labor_force'].mean()
# type(x)
differenced_2 = labor_force_1.diff(1)
# trim off the first year of empty data
differenced_2 = differenced_2[1:]
# plot differenced dataset
differenced_2.plot()

plot_acf(differenced_2)
pyplot.show()

# Merge these lagged variables with the train and test

x_1 = x_1.shift(1)
x_1 = x_1.to_frame().reset_index(level=0)
print x_1.head()
rub_1 = rub_1.shift(1)
rub_1 = rub_1.to_frame().reset_index(level=0)
print rub_1.head()
labor_force_1 = labor_force_1.shift(1)
labor_force_1 = labor_force_1.to_frame().reset_index(level=0)
print labor_force_1.head()


yrmth = pd.merge(x_1, rub_1, how = 'left', on = 'year_month')

yrmth = pd.merge(yrmth, labor_force_1, how = 'left', on = 'year_month')
yrmth

# Same for test

oil_test = test.groupby('year_month')['oil_urals'].mean()
# type(x)
differenced = oil_test.diff(1)
# trim off the first year of empty data
differenced = differenced[1:]
# plot differenced dataset

rub_test = test.groupby('year_month')['usdrub'].mean()
# type(x)
differenced_t1 = rub_test.diff(1)
# trim off the first year of empty data
differenced_t1 = differenced_t1[1:]
# plot differenced dataset

labor_force_test = test.groupby('year_month')['labor_force'].mean()
# type(x)
differenced_t2 = labor_force_test.diff(1)
# trim off the first year of empty data
differenced_t2 = differenced_t2[1:]

# Make into dataframe
oil_test = oil_test.shift(1)
oil_test = oil_test.to_frame().reset_index(level=0)
print oil_test.head()

rub_test = rub_test.shift(1)
rub_test = rub_test.to_frame().reset_index(level=0)
print rub_test.head()

labor_force_test = labor_force_test.shift(1)
labor_force_test = labor_force_test.to_frame().reset_index(level=0)
print labor_force_test.head()

yrmth_test = pd.merge(oil_test, rub_test, how = 'left', on = 'year_month')

yrmth_test = pd.merge(yrmth_test, labor_force_test, how = 'left', on = 'year_month')
yrmth_test

# Join with training and test
train = pd.merge(train, yrmth, how = 'left', on = 'year_month')
test = pd.merge(test, yrmth_test, how = 'left', on = 'year_month')

test.head()

# Find change of lag (_y) vs current (_x)
train['delta_oil'] = (train['oil_urals_y'] - train['oil_urals_x']) / train['oil_urals_y']

train.loc[:,['year_month','oil_urals_x','oil_urals_y', 'delta_oil']]

train['delta_usdrub'] = (train['usdrub_y'] - train['usdrub_x']) / train['usdrub_y']
train.loc[:,['year_month','usdrub_x','usdrub_y', 'delta_usdrub']]

train['delta_labor_force'] = (train['labor_force_y'] - train['labor_force_x']) / train['labor_force_y']
train.loc[:,['year_month','labor_force_x','labor_force_y', 'delta_labor_force']]

# Do same for test
test['delta_oil'] = (test['oil_urals_y'] - test['oil_urals_x']) / test['oil_urals_y']
test['delta_usdrub'] = (test['usdrub_y'] - test['usdrub_x']) / test['usdrub_y']
test['delta_labor_force'] = (test['labor_force_y'] - test['labor_force_x']) / test['labor_force_y']

train = train.fillna(0)

test = test.fillna(0)

cols = range(3,48)

train.head()

from sklearn.ensemble import RandomForestRegressor

y = np.log(train[[0]])
X = train.iloc[:,cols]
x_test = test.iloc[:,range(2,47)]
# print X
# Note I excluded timestamp

model = RandomForestRegressor(n_estimators=1000, oob_score=True)
# Train the model using the training sets and check score
model.fit(X, y.values.ravel())
#Predict Output
predicted= model.predict(x_test)

model 

predicted

y.values.ravel()

from sklearn.metrics import roc_auc_score
#oob_error = 1 - model.oob_score_
#print model.oob_prediction_
model.score(X, y)
#print "AUC - ROC : ", roc_auc_score(y.values, model.score(X,y))

model.oob_prediction_.ravel()

y.values.ravel()

model.feature_importances_ 

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

X.iloc[:,27]

print list(X)[0]
print list(X)[27]
print list(X)[15]
print list(X)[22]
print list(X)[21]
print list(X)[19]
print list(X)[17]
print list(X)[16]
print list(X)[26]
print list(X)[18]
print list(X)[20]
print list(X)[37]

X2 = train.loc[:, ['month','year', 'day','year_month','num_room','kremlin_km','cafe_count_5000',
                    'metro_min_avto','additional_education_km','build_year','oil_urals_y', 'industrial_km',
                   'state','usdrub_y', 'labor_force_y','delta_oil', 'delta_usdrub','delta_labor_force']]


y2 = np.log(train[[0]])
x2_test = test.loc[:, ['month','year', 'day','year_month','num_room','kremlin_km','cafe_count_5000',
                    'metro_min_avto','additional_education_km','build_year','oil_urals_y', 'industrial_km',
                   'state','usdrub_y', 'labor_force_y','delta_oil', 'delta_usdrub','delta_labor_force']]


model2 = RandomForestRegressor(n_estimators=1000, oob_score=True, verbose = 1)
# Train the model using the training sets and check score
model2.fit(X2, y2.values.ravel())
#Predict Output
#predicted2 = model2.predict(x2_test)

model2.fit(X2, y2.values.ravel())

model2.score(X2, y2)

importances2 = model2.feature_importances_
std = np.std([tree.feature_importances_ for tree in model2.estimators_],
             axis=0)
indices = np.argsort(importances2)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances2[indices[f]]))

predicted2 = model2.predict(x2_test)

type(predicted2)

predicted2


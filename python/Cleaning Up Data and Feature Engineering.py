#Import modules:
import pandas as pd
import numpy as np
import scipy as sp
import sklearn

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

#attach the okrug region
okurg_df = pd.read_csv("okurg.csv")
trainsm_df = pd.merge(trainsm_df, okurg_df, how='left', on='sub_area')
testsm_df = pd.merge(testsm_df, okurg_df, how='left', on='sub_area')

#Checking out the data
dtype_df = trainsm_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

dtype_df2 = testsm_df.dtypes.reset_index()
dtype_df2.columns = ["Count", "Column Type"]
dtype_df2.groupby("Column Type").aggregate('count').reset_index()

from sklearn import model_selection, preprocessing
for f in trainsm_df.columns:
    if trainsm_df[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(trainsm_df[f].values.astype('str')) + list(testsm_df[f].values.astype('str')))
        trainsm_df[f] = lbl.transform(list(trainsm_df[f].values.astype('str')))
        testsm_df[f] = lbl.transform(list(testsm_df[f].values.astype('str')))

missing = trainsm_df.isnull().sum()
print(missing.to_string())
print len(trainsm_df)

trainsm_df['year'] = pd.DatetimeIndex(trainsm_df['timestamp']).year

trainsm_df.describe()

missing2 = testsm_df.isnull().sum()
print(missing2.to_string())
print len(testsm_df)

#fix train
train_nofloor = trainsm_df.dropna(subset = ['floor'])
train_clean = train_nofloor.dropna(subset = ['metro_min_walk'])

testsm_df.describe()

testsm_df.loc[testsm_df['max_floor'] < testsm_df['floor']]

#fix test
test_clean = testsm_df.dropna(subset = ['metro_min_walk'])

#Make a new column that is just the sq_mter, to get rid of issues with Life Square not existing or being incorrect
train_clean.loc[:, 'sq_metr'] = train_nofloor.loc[:, ['full_sq','life_sq']].max(axis=1)
test_clean.loc[:, 'sq_metr'] = test_clean.loc[:, ['full_sq','life_sq']].max(axis=1)

train_clean = train_clean.dropna(axis=1, how='any')
test_clean = test_clean.dropna(axis=1, how='any')
test_clean = test_clean.drop(["max_floor", "material", "num_room", "kitch_sq"], axis=1)

train_clean.describe()

#find where the meters does not make sense (20 instances)
train_clean.loc[train_clean['sq_metr'] < 6]
train_clean = train_clean.drop(train_clean[train_clean['sq_metr'] < 6].index)

test_clean.describe()

#drop one observation in the test as well
test_clean.loc[test_clean['sq_metr'] < 6]
test_clean = test_clean.drop(test_clean[test_clean['sq_metr'] < 6].index)

#when is sq meter too large?
train_clean.loc[train_clean['sq_metr'] > 300]

#Imputation to make the square footage make sense
train_clean.loc[train_clean['sq_metr'] > 1000, 'sq_metr'] = train_clean.loc[train_clean['sq_metr'] > 1000, 'sq_metr']/100
train_clean.loc[train_clean['sq_metr'] > 310, 'sq_metr'] = train_clean.loc[train_clean['sq_metr'] > 310, 'sq_metr']/10

#checking it:
train_clean.loc[train_clean['sq_metr'] > 300]

#the two (9230 and 16727) are very large units that were very expensive

#when is sq meter too large for the test data?
test_clean.loc[test_clean['sq_metr'] > 300]
#looking at the full_sq for the first three, just looks like they were shifted 1 decmimal -- divide by 10

test_clean.loc[test_clean['sq_metr'] > 310, 'sq_metr'] = test_clean.loc[test_clean['sq_metr'] > 310, 'sq_metr']/10

test_clean.loc[test_clean['sq_metr'] > 300]

test_clean.describe()

train_clean.describe()

list(train_clean)

#Population Density (will be the same throughout each SubArea)
train_clean["pop_density"] = train_clean["raion_popul"] / train_clean["area_m"].astype("float")
test_clean["pop_density"] = test_clean["raion_popul"] / test_clean["area_m"].astype("float")

#Ratio of elder population (will be the same throughout each SubArea)
train_clean["elder_ratio"] = train_clean["ekder_all"] / (train_clean["young_all"] + train_clean["work_all"] + train_clean["ekder_all"]).astype("float")
test_clean["elder_ratio"] = test_clean["ekder_all"] / (test_clean["young_all"] + test_clean["work_all"] + test_clean["ekder_all"]).astype("float")

#Ratio of under 18 population (will be the same throughout each SubArea)
train_clean["youth_ratio"] = train_clean["young_all"] / (train_clean["young_all"] + train_clean["work_all"] + train_clean["ekder_all"]).astype("float")
test_clean["youth_ratio"] = test_clean["young_all"] / (test_clean["young_all"] + test_clean["work_all"] + test_clean["ekder_all"]).astype("float")

#Ratio of number of preschool aged children to number of preschools (will be the same throughout each SubArea)
#train_clean["preschool_ratio"] = train_clean["children_preschool"] / train_clean["preschool_education_centers_raion"].astype("float")
#test_clean["preschool_ratio"] = test_clean["children_preschool"] / test_clean["preschool_education_centers_raion"].astype("float")

#this doesn't look like it worked...

test_clean.describe()

features = ['id',
 'timestamp',
 'floor',
 'product_type',
 'sub_area',
 'metro_min_walk',
 'kindergarten_km',
 'park_km',
 'kremlin_km',
 'oil_chemistry_km',
 'nuclear_reactor_km',
 'big_market_km',
 'market_shop_km',
 'detention_facility_km',
 'public_healthcare_km',
 'university_km',
 'workplaces_km',
 'preschool_km',
 'big_church_km',
 'oil_urals',
 'cpi',
 'ppi',
 'eurrub',
 'brent',
 'average_provision_of_build_contract_moscow',
 'micex',
 'mortgage_rate',
 'rent_price_4+room_bus',
 'sd_oil_yearly',
 'sd_cpi_yearly',
 'sd_ppi_yearly',
 'sd_eurrub_yearly',
 'sd_brent_yearly',
 'sd_micex_yearly',
 'sd_mortgage_yearly',
 'sd_rent_yearly',
 'okurg_district',
 'sq_metr',
 'pop_density',
 'elder_ratio',
 'youth_ratio',
 'price_doc']

train_trial1 = train_clean[features]
test_trial1 = test_clean[features[:-1]]

train_trial1.shape

train_trial1.to_csv('trial_brandy.csv', index = False)

test_trial1.to_csv('test_brandy.csv', index = False)





from sklearn import naive_bayes

## separate the predictors and response in the training data set
x = np.array(train_trial1.iloc[:, 2:41])
y = np.ravel(train_trial1.iloc[:, 41:42])

x
y

#mnb = naive_bayes.MultinomialNB()
#mnb.fit(x, y)
#print("The score of multinomial naive bayes is: %.4f" %mnb.score(x, y))



X = train2.drop(["id", "price_doc"], axis = 1)

Y = train2["price_doc"]

model = sm.OLS(Y, X)
X = sm.add_constant(X)
results = model.fit()
print(results.summary())






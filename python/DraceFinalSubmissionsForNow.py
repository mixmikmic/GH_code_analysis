import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedKFold
from sklearn import preprocessing
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import re
import xgboost as xgb
color = sns.color_palette()
get_ipython().magic('matplotlib inline')
from xgboost import XGBClassifier

df_train = pd.read_json('final_train.json')
df_test = pd.read_json('final_test.json')

#original_train = pd.read_json('train.json')
#original_test = pd.read_json('test.json')

#df_train = df_train.drop(['manager_id', 'building_id'], 1)
#df_test = df_test.drop(['manager_id', 'building_id'], 1)

#original_train = original_train[['manager_id', 'building_id', 'listing_id']]
#original_test = original_test[['manager_id', 'building_id', 'listing_id']]

#df_train['listing_id'].nunique()

#df_train = df_train.merge(original_train, how = 'left', left_on = 'listing_id', right_on ='listing_id')

#df_test = df_test.merge(original_test, how = 'left', left_on = 'listing_id', right_on = 'listing_id')

#df_train.to_json('C:\\Users\\Drace\\Documents\\final_train.json')
#df_test.to_json('C:\\Users\\Drace\\Documents\\final_test.json')

drace_df = df_train

categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if df_train[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df_train[f].values) + list(df_test[f].values))
            df_train[f] = lbl.transform(list(df_train[f].values))
            df_test[f] = lbl.transform(list(df_test[f].values))

feats_used = ['bathrooms', 'bedrooms', 'latitude', 'longitude',"display_address", "manager_id", "building_id", "street_address",              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words',             'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets','Price_P_Room',              'nofee', 'has_phone',                'dist_to_nearest_college', 'weekday_created', 'large_space','created', 'created_hour']
x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
rf = RandomForestClassifier(n_estimators=1600, max_depth= 17, min_samples_leaf= 1, oob_score=True)
rf.fit(x_train[feats_used], y_train)
y_pred0 = rf.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred0)

feats_used = ['bathrooms', 'bedrooms', 'latitude', 'longitude',"display_address", "manager_id", "building_id", "street_address",              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words',             'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets','Price_P_Room',              'nofee', 'has_phone',                'dist_to_nearest_college', 'weekday_created', 'large_space','created', 'created_hour']
x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
xgb0 = XGBClassifier(n_estimators=2000)
xgb0.fit(x_train,y_train)
y_pred1 = xgb0.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred1)

#some baseline features according to some good rule of thumbs for gbm params

feats_used = ['bathrooms', 'bedrooms', 'latitude', 'longitude',"display_address", "manager_id", "building_id", "street_address",              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words',             'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets','Price_P_Room',              'nofee', 'has_phone',                'dist_to_nearest_college', 'weekday_created', 'large_space','created', 'created_hour']
x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
gbm0 = GradientBoostingClassifier(n_estimators=2000, max_features=5, subsample= 0.8)
gbm0.fit(x_train,y_train)
y_pred2 = gbm0.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred2)

ef = ExtraTreesClassifier(n_estimators=2000, max_depth= 25, min_samples_leaf= 5, criterion= 'entropy')
ef.fit(x_train, y_train)
y_pred3 = ef.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred3)

#increase depth lead to less log_loss, entropy, gini seems to work the same

def manager_skill(df):
#new var to create
    new_var = 'manager_id'#'manager_id_encoded'
#response var
    resp_var = 'interest_level'
# Step 1: create manager_skill ranking from training set:
    train_df = pd.read_json("train.json") # upload training scores => test data cannot create a rank skill
    temp = pd.concat([train_df[new_var], pd.get_dummies(train_df[resp_var])], axis = 1).groupby(new_var).mean()
    temp.columns = ['high_frac','low_frac', 'medium_frac']
    temp['count'] = train_df.groupby(new_var).count().iloc[:,1]
    temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
# Step 2: Fill working dataset (e.g. test set) with ranking figures using left-merge to match original_id and temp dfs
    mean_manager_skill = np.mean(temp['manager_skill'])

    skill_df = pd.merge(df[['manager_id']],temp[['manager_skill']],how='left', left_on='manager_id', right_index=True)[['manager_skill']]

    skill_df.fillna(mean_manager_skill, inplace=True)
    df['manager_skill'] = list(skill_df['manager_skill'])
    return df

manager_skill(drace_df)
manager_skill(df_test)

pred_logit = ['bathrooms', 'bedrooms', 'price_vs_median_72', 'n_log_price','n_expensive',
              'n_no_photo', 'num_photos','n_num_keyfeat_score', 'num_description_words', 'has_phone',
              'manager_skill','dist_to_nearest_tube', 'subway', 'weekday_created', 'allow_pets', 'amount_of_caps',
             'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman', 'nofee']

## separate the predictors and response:
from sklearn import metrics 
# Training:
x = drace_df[pred_logit]
y = drace_df['interest_level']

from sklearn import linear_model
logit_1 = linear_model.LogisticRegression()
logit_1.fit(x, y)

y_pred = logit_1.predict(x)
y_pred_p = logit_1.predict_proba(x)
print 'Multi Class Log_loss:', metrics.log_loss(y,y_pred_p)

from sklearn.externals import joblib
joblib.dump(gbm0, 'gbm_model0.pkl')
joblib.dump(rf, 'rf_model0.pkl')
joblib.dump(xgb0, 'xgb0_model2.pkl')

x_Test = df_test[feats_used]
rf_preds = rf.predict(x_Test)
gbm_preds = gbm0.predict(x_Test)
ef_preds = ef.predict(x_Test)

x_Test = df_test[feats_used]
eclf = EnsembleVoteClassifier(clfs=[rf, gbm0, xgb0, logit_1, ef], weights=[1, 1, 1, 1, 1], voting='hard')

eclf.fit(x_train, y_train)
y_predE = eclf.predict_proba(x_test[feats_used])
log_loss(y_test, y_predE)

feats_used = ['bathrooms', 'bedrooms', 'latitude', 'longitude',              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words',              'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets','Price_P_Room',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',                'dist_to_nearest_college', 'weekday_created', 'large_space', 'created', 'created_hour']
x_Test = df_test[feats_used]
submissioneclf = pd.DataFrame(eclf.predict_proba(x_Test[feats_used]))
submissioneclf = pd.concat([df_test.reset_index(drop=True), submissioneclf], axis=1)
submissioneclf.rename(columns={0: 'high', 1: 'low', 2: 'medium'}, inplace=True)
submissioneclf = submissioneclf[['listing_id', 'high', 'medium', 'low']]
submissioneclf.head()

submissioneclf.to_csv('C:\\Users\\Drace\\Documents\\submissionECLF.csv', index = False)

#can we meet default xgboost??
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

feats_used = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'manager_id', 'building_id', 'display_address', 'street_address',              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words',             'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets','Price_P_Room',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',                'dist_to_nearest_college', 'weekday_created', 'large_space','created', 'created_hour']
train_X = df_train[feats_used]
test_X = df_test[feats_used]

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(df_train['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

preds, model = runXGB(train_X, train_y, test_X, num_rounds=400)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = df_test.listing_id.values
out_df.to_csv("xgb_baseline.csv", index=False)

gbm2 = GradientBoostingClassifier(n_estimators=80, min_samples_split=241, max_depth=15)
gbm2.fit(x_train,y_train)
y_predgbm2 = gbm2.predict_proba(x_test[feats_used])
log_loss(y_test, y_predgbm2)

rf2
ef2

feats_used = ['bathrooms', 'bedrooms', 'latitude', 'longitude',"display_address", "manager_id", "building_id", "street_address",              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words',             'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets','Price_P_Room',              'nofee', 'has_phone',                'dist_to_nearest_college', 'weekday_created', 'large_space','created', 'created_hour']
x_Test = df_test[feats_used]
submissionRF = rf.predict_proba(x_Test)

submissionRF = pd.DataFrame(submissionRF)

submissionRF = pd.concat([df_test.reset_index(drop=True), submissionRF], axis=1)

submission_rf = submissionRF[['listing_id',0,1, 2]]

submission_rf.rename(columns={0: 'high', 1: 'low', 2: 'medium'}, inplace=True)
submission_rf = submission_rf[['listing_id', 'high', 'medium', 'low']]

submission_rf.head()

submission_rf.to_csv('C:\\Users\\Drace\\Documents\\submissionRF.csv', index = False)

submission_rf.columns

df_test['price_vs_median_72'] = df_test['price']/df_test['median_72']

feats_used = ['bathrooms', 'bedrooms', 'latitude', 'longitude',"display_address", "manager_id", "building_id", "street_address",              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words',             'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets','Price_P_Room',              'nofee', 'has_phone',                'dist_to_nearest_college', 'weekday_created', 'large_space','created', 'created_hour']
x_Test = df_test[feats_used]
submissiongbm0 = pd.DataFrame(gbm0.predict_proba(x_Test[feats_used]))
submissiongbm0 = pd.concat([df_test.reset_index(drop=True), submissiongbm0], axis=1)
submissiongbm0.rename(columns={0: 'high', 1: 'low', 2: 'medium'}, inplace=True)

submissiongbm0 = submissiongbm0[['listing_id', 'high', 'medium', 'low']]
submissiongbm0.head()

submissiongbm0.to_csv('C:\\Users\\Drace\\Documents\\submissiongbm0.csv', index = False)

feats_used = ['bathrooms', 'bedrooms', 'latitude', 'longitude',"display_address", "manager_id", "building_id", "street_address",              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words',             'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets','Price_P_Room',              'nofee', 'has_phone',                'dist_to_nearest_college', 'weekday_created', 'large_space','created', 'created_hour']
x_Test = df_test[feats_used]
submissionxgb0 = pd.DataFrame(xgb0.predict_proba(x_Test[feats_used]))
submissionxgb0 = pd.concat([df_test.reset_index(drop=True), submissionxgb0], axis=1)
submissionxgb0.rename(columns={0: 'high', 1: 'low', 2: 'medium'}, inplace=True)
submissionxgb0 = submissionxgb0[['listing_id', 'high', 'medium', 'low']]
submissionxgb0.head()

submissionxgb0.to_csv('C:\\Users\\Drace\\Documents\\submissionxgb0.csv', index = False)

from sklearn.grid_search import GridSearchCV
param_test1 = {'n_estimators':range(1000,2001,200)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_depth= 10, min_samples_leaf= 1, n_jobs=2, max_features='auto'),                        param_grid = param_test1, scoring='log_loss',n_jobs=2,iid=False, cv=5)
gsearch1.fit(drace_df[feats_used],y)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

from sklearn.grid_search import GridSearchCV
max_depth_range = range(1, 11)
leaf_range = range(1, 11)
param_grid2 = dict(max_depth=max_depth_range, min_samples_leaf=leaf_range)
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 80, n_jobs=2, max_features='auto'),                        param_grid = param_grid2, scoring='log_loss',n_jobs=2,iid=False, cv=5)
gsearch2.fit(drace_df[feats_used],y)

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

from sklearn.grid_search import GridSearchCV
param_test3 = {'max_depth':range(11, 21)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 1600, min_samples_leaf= 1, n_jobs=2, max_features='auto'),                        param_grid = param_test3, scoring='log_loss',n_jobs=2,iid=False, cv=5)
gsearch3.fit(drace_df[feats_used],y)

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=100,learning_rate=1)
adb.fit(x_train, y_train)
y_pred4 = adb.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred4)

#not using this, unsure of why this model is so weak


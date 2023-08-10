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
import time

cd /Users/williamzhou/Downloads/RussianHousing-master

train=pd.read_csv('./data/processed/Clean0517/train_clean_shu_0517.csv')
test = pd.read_csv('./data/processed/Clean0517/test_clean_shu_0517.csv')
macro = pd.read_csv('./data/raw/macro.csv')
# latlon = pd.read_csv('./data/external/sub_area_lon_lat.csv')
feature_importance = pd.read_csv('/Users/williamzhou/Documents/github/RussianHousing/feature engineering/feature_importance.csv')

train['timestamp'] = pd.to_datetime(train['timestamp'])
macro['timestamp'] = pd.to_datetime(macro['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])

train['label']='train'
test['label']='test'

print ('train shape',train.shape)
print ('test shape',test.shape)
print ('feature_importance shape', feature_importance.shape)

time_laps_dict ={
#                     'usdrub':0#,
                 'micex_cbi_tr':0,
                 'micex_rgbi_tr':0#,
#                  'deposits_value':30
#                  'ppi':90,
#                  'cpi':300
                }

def merge_macro_feature(dicts,macro,df):
    
    for item in dicts:
        macro_timeshift=macro.copy()
        macro_timeshift['timestamp']=macro.timestamp+timedelta(days=dicts[item])
        df = pd.merge(df,macro_timeshift[['timestamp',item]],on='timestamp',how='left')
    print('data shape:',df.shape)
    return(df)

train =merge_macro_feature(time_laps_dict,macro,train)
test=merge_macro_feature(time_laps_dict,macro,test)

#### Group frequent area
freq_area = np.array(train.loc[:, 'sub_area'].value_counts()[:2].index)
train.loc[~train['sub_area'].isin(freq_area), 'sub_area'] = 'other'
test.loc[~test['sub_area'].isin(freq_area), 'sub_area'] = 'other'
print ('subarea are {}'.format(train.sub_area.unique()))

train.loc[train['full_sq'].isnull(),'full_sq']=train['full_sq'].median()
test.loc[test['full_sq'].isnull(),'full_sq']=test['full_sq'].median()
train['price_full_sq']=train['price_doc']/train['full_sq']
train['price_full_sq']=train['price_full_sq'].astype('int64')

# train= median_price_sqm_by_ID('ID_metro',train)
# test= median_price_sqm_by_ID('ID_metro',test)
print ('train shape',train.shape)
print ('test shape',test.shape)

y_train=train[['price_full_sq','sub_area']]
x_train= train.drop(['price_doc','price_full_sq'],axis=1)
test_id = test[['id','sub_area']]
x_test = test.drop(['id'],axis=1)

print('x_train shape:',x_train.shape)
print('y_train shape:',y_train.shape)
print('x_test shape:',x_test.shape)
print('test_id shape:',test_id.shape)

df_all = pd.concat([x_train,x_test])

x_train=df_all.loc[df_all.label=='train',:]
x_train.shape

IDs = ['ID_big_road1','ID_big_road2',
        'ID_bus_terminal','ID_metro',
       'ID_railroad_station_avto',
       'ID_railroad_terminal']

def to_object(ID,df):
    df[ID] = df[ID].astype(object)
    return(df)

df_all=to_object(IDs,df_all)

    
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

for c in df_all.columns:
    if df_all[c].dtype == 'object' and c!='label' and c!='sub_area':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values)) 
        df_all[c] = lbl.transform(list(df_all[c].values))

df_all=df_all.drop(['timestamp'],axis =1 )

x_train = df_all.loc[df_all.label=='train',:]
x_test = df_all.loc[df_all.label=='test',:]

print('x_train shape',x_train.shape)
print('x_test shape',x_test.shape)
print('y_train shape',y_train.shape)

final_features = pd.read_csv('./cv_output/final_feature.csv')
train_feature = list(final_features.iloc[:,0])
print('Training XGBoost with %d features'%len(train_feature))

subareas=x_train.sub_area.unique()
print('Subareas has %d areas'%len(subareas))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0
}

def median_missing_imputation(df):
    for cols in df.columns:
            df.loc[pd.isnull(df[cols]),cols] = np.median(df.loc[pd.isnull(df[cols])==False,cols])

    return(df)

get_ipython().run_cell_magic('time', '', "sub=pd.DataFrame()\nfor area in subareas:\n# for area in ['Strogino']:\n    x_train_subset = x_train.loc[x_train.sub_area==area,train_feature].reset_index(drop=True)\n    x_test_subset = x_test.loc[x_test.sub_area==area,train_feature].reset_index(drop=True)\n    y_train_subset = y_train.loc[x_train.sub_area==area,'price_full_sq'].reset_index(drop=True)\n    test_id_subset =test_id.loc[test_id.sub_area==area,'id'].reset_index(drop=True)\n    \n#     missing imputation using on subarea median\n#     x_train_subset = median_missing_imputation(x_train_subset)\n#     x_test_subset = median_missing_imputation(x_test_subset)\n    print('Finish missing imputation.')\n    dtrain_subset = xgb.DMatrix(x_train_subset, y_train_subset)\n    dtest_subset =  xgb.DMatrix(x_test_subset)\n    print('{} x_train_subset shape {}'.format(area,x_train_subset.shape))\n    print('{} y_test_subset shape {}'.format(area,y_train_subset.shape))\n    print('{} x_test_subset shape {}'.format(area,x_test_subset.shape))\n    print('{} test_id_subset shape {}'.format(area,test_id_subset.shape))\n    print('Ready! Start training {}......'.format(area))\n    \n    cv_output = xgb.cv(xgb_params, dtrain_subset, \n                   num_boost_round=1000, \n                   early_stopping_rounds=20,\n                   verbose_eval=50, show_stdv=False)\n    cv_output.to_csv('./train_by_subarea/{}.csv'.format(area))\n    print('{}.csv successfully saved!'.format(area))\n    num_boost_rounds = len(cv_output)\n    print('Best number of iteration for {} model: {})'.format(area,num_boost_rounds))\n    model = xgb.train(dict(xgb_params, silent=0), dtrain_subset, num_boost_round= num_boost_rounds)\n    print('Finish XGBoost training')\n    print('Predicting........')\n    y_predict = model.predict(dtest_subset)\n    y_predic_all_sq = (y_predict)*x_test_subset['full_sq']\n#     print(y_predic_all_sq)\n#     print(test_id.id[test_id.sub_area==area])\n    df_sub = pd.DataFrame({'id':test_id_subset , 'price_doc': y_predic_all_sq})\n    print([x for x in df_sub])\n    sub = pd.concat([sub,df_sub])\n    print('Successfully trained {}'.format(area))\n    print('----------------------------------------------------------')\n    print('----------------------------------------------------------')\nprint('The prediction dataset should have 7662 rows of data, your dataset has {} rows of data'.format(len(sub)))\n\nsub.sort_values(by='id').reset_index(drop=True).to_csv('./train_by_subarea/sub_{}.csv'.format(time.ctime()),index=False)\nprint('Successfully saved submission file! ')")




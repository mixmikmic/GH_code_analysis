import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime


cd /Users/williamzhou/Desktop/russian_real_estate

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
macro = pd.read_csv('./macro.csv')
id_test = test.id

# y_train = np.log1p(train["price_doc"]) # log1p transformation may hurt performance
y_train = train["price_doc"]
x_train = train.drop(["id", "price_doc"], axis=1)
x_test = test.drop(["id"], axis=1)

num_train = len(train)
df_all = pd.concat([x_train,x_test])

# change ID variable to categorical variables
df_all[['ID_big_road1','ID_big_road2',
        'ID_bus_terminal','ID_metro',
       'ID_railroad_station_avto',
       'ID_railroad_terminal']]=df_all[['ID_big_road1','ID_big_road2',
                                        'ID_bus_terminal','ID_metro',
                                       'ID_railroad_station_avto',
                                       'ID_railroad_terminal']].astype(object)

# Create new features based on timestamp
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)




# df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
# df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
# df_all['life_full_sq_ratio'] = df_all['life_sq']/df_all['full_sq']
# df_all['avg_room_sq'] = (df_all['life_sq']-df_all['kitch_sq'])/df_all['num_room']

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values)) 
        df_all[c] = lbl.transform(list(df_all[c].values))

df_all=df_all.drop(['timestamp'],axis =1 )

x_train = df_all.iloc[:num_train,:]
x_test = df_all.iloc[num_train:,:]

print('x_train shape',x_train.shape)
print('x_test shape',x_test.shape)
print('y_train shape',y_train.shape)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

get_ipython().run_cell_magic('time', '', "cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,\n    verbose_eval=50, show_stdv=False)\ncv_output[['train-rmse-mean', 'test-rmse-mean']].plot()")

get_ipython().run_cell_magic('time', '', 'num_boost_rounds = len(cv_output)\nprint(num_boost_rounds)\nmodel = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)')

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

importance = model.get_fscore()
importance = pd.DataFrame(sorted(importance.items()),columns =['feature','fscore'])
importance = importance.sort_values(by='fscore',ascending=False).reset_index(drop=True)
importance_features = list(importance.loc[:,'feature'])
importance_features

y_predict = model.predict(dtest)
# y_predict = np.exp(y_predict) - 1
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
df_sub.to_csv('./sub.csv', index=False)

importance.to_csv('./feature_importance.csv')

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
rmse_result_dict={}

for i in [95]:
    # create subset of df_all
    df_all_subset = df_all.loc[:,importance_features[:i]]
#     prepare data
    x_train_subset = df_all_subset.iloc[:num_train,:]
    dtrain_subset = xgb.DMatrix(x_train_subset, y_train)
#     Train model / Cross validation 
    cv_output = xgb.cv(xgb_params, dtrain_subset, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
    print('finish',i)
#     save result
    rmse_result_dict[i]=cv_output.iloc[-1,:]
    

test_rmse_mean=[]
top_n_features=[]
for i in range(80,111,5):
    top_n_features.append(i)
    test_rmse_mean.append(rmse_result_dict[i]['test-rmse-mean'])

    

# rmse=pd.DataFrame(zip(top_n_features,(x/10**3 for x in test_rmse_mean)),columns=['Top_n_features','test_rmse_mean'])
rmse=pd.DataFrame(zip(top_n_features,test_rmse_mean),columns=['Top_n_features','test_rmse_mean'])
fig, ax = plt.subplots()
plt.plot(rmse.Top_n_features,rmse.test_rmse_mean)
ax.set(xlabel='Top n important features',
       ylabel='Test_rmse_mean',
       title = 'Choose n top features to get best CV-rmse')

plt.show()

num_boost_round = model.best_iteration
model = xgb.train(dict(xgb_params, silent=0), dtrain_subset, num_boost_round=num_boost_round)
fig, ax = plt.subplots(1, 1, figsize=(8, 40))
xgb.plot_importance(model, max_num_features=200, height=0.5, ax=ax)
plt.show()

x_test_subset = df_all_subset.iloc[num_train:,:]
dtest_subset = xgb.DMatrix(x_test_subset)

y_pred = model.predict(dtest_subset)
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('./sub.csv', index=False)




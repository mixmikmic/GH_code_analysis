import pandas as pd
from sklearn import preprocessing
import numpy as np
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '%.1f' % x)

train = pd.read_csv('../Chase/data/clean_train_all_cols_outliers_removed_chase.csv', parse_dates=['timestamp'], index_col="id")  
test = pd.read_csv('../Chase/data/clean_test_all_cols_chase.csv', parse_dates=['timestamp'], index_col="id")

def getKremlinGroup(df, id):
    """ returns the group that are the same distance from the kremlin"""
    x = df.loc[id,'kremlin_km']
    return df.loc[df.kremlin_km==x,:]

train_index = train.index.tolist()
test_index = test.index.tolist()

cols = ['life_sq','full_sq','floor','max_floor','kitch_sq','sub_area','kremlin_km','price_doc','timestamp']

test['price_doc'] = np.nan

df = pd.concat([train[cols].copy(),
                test[cols].copy()],
               ignore_index=False)

df['month'] = df.timestamp.dt.month.astype(object)

macro = pd.read_csv('../Chase/data/macro_chase.csv')
macro['quarter'] = pd.PeriodIndex(macro['Unnamed: 0'], freq='Q').strftime('Q%q-%y')
df['quarter'] = pd.PeriodIndex(df['timestamp'], freq='Q').strftime('Q%q-%y')

df = pd.merge(df,macro[['quarter','nominal_index']], how="left", on="quarter").reset_index(drop=True).set_index(df.index)

df['kitch_to_life'] = df.kitch_sq / df.life_sq
df['life_to_full'] = df.life_sq / df.full_sq
df['bld_type'] = 'med_rise'
df.loc[df.max_floor <= 5,'bld_type'] = 'low_rise'
df.loc[df.max_floor >= 17,'bld_type'] = 'high_rise'
df['walk_up_penalty'] = 0
df.loc[(df.floor>4) & (df.max_floor < 6),'walk_up_penalty'] = 1 

df['price_doc'] = df.price_doc / df.nominal_index
df['price_full'] = df.price_doc / df.full_sq
df['log_price'] = np.log(df.price_doc)
# df['price_doc'] = df.price_doc / 1000

from sklearn import linear_model
from sklearn.model_selection import KFold, cross_val_score

ols = linear_model.LinearRegression()

# cols to drop
# drop_cols = ['timestamp','price_doc','nominal_index','adj_price_doc','price_full','log_price','price_full']
cols = ['full_sq','floor','sub_area','kremlin_km','month']
lm_data = df[cols].copy()

df_obj = lm_data.select_dtypes(include=['object'])
df_num = lm_data.select_dtypes(exclude=['object'])


dummies = pd.get_dummies(df_obj)
df_all = pd.concat([df_num,dummies],axis=1)

x_train = df_all.loc[train_index]

y_train = df.loc[train_index,'log_price']

x_test = df_all.loc[test_index,:]

ols.fit(x_train,y_train)
print('R^2: %.2f' % ols.score(x_train, y_train))
# df.log_price

df.loc[test_index,'price_doc'] = np.exp(ols.predict(x_test)) * df.loc[test_index,'nominal_index']

df['price_full'] = df.price_doc / df.full_sq

cols = ['price_doc','full_sq','price_full']
sub = df.loc[test_index,cols]

sub.loc[37686,'price_doc'] = sub.loc[37686,'full_sq'] * 225000
sub.loc[[34670,32941],'price_doc'] = sub.loc[[34670,32941],'full_sq'] * 275000
sub.loc[33974,'price_doc'] = sub.loc[33974,'full_sq'] * 175000

sub[(sub.price_full > 200000) & (sub.full_sq<15)]
# sub.sort_values('price_full',ascending=False)

# kaggle score 0.34
sub['price_doc'].to_frame().to_csv('../Chase/submissions/simple_linear_052716.csv')















cv_scores = cross_val_score(ols, x_train, y_train, cv=10)
print cv_scores

ols.predict(x_test)

cv_scores = cross_val_score(ols, x_train, y_train, cv=10)
print cv_scores

test1 = pd.DataFrame({'price_doc': ols.predict(x_test)},index=test_index)

test['price_doc'] = ols.predict(x_test)



from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


pipe  =  make_pipeline(MinMaxScaler(), Ridge())
param_grid = {'ridge__alpha': [100,10,1,0.1,0.01,0.001,0.0001,0]}
grid =  GridSearchCV(pipe, param_grid, cv=5)
lm_predictions = grid.fit(x_train, y_train)

# print lm_predictions.predict(x_train)
print lm_predictions.best_score_

from sklearn import preprocessing 



cols = ['timestamp','price_doc','nominal_index','adj_price_doc','price_full','log_price','price_full']

pipe  =  make_pipeline(MinMaxScaler(), Ridge())
param_grid = {'ridge__alpha': [100,10,1,0.1,0.01,0.001,0.0001,0]}
grid =  GridSearchCV(pipe, param_grid, cv=5)
grid.fit(x_train, y_train)

sub = grid.predict(x_test)

sub = pd.DataFrame({'id': test_index, 'price_doc':sub})

sub.loc[:,'nominal_index'] = df.loc[test_index,'nominal_index'].values

'%f' % 1.128899e+08

sub.price_doc = sub.price_doc * sub.nominal_index

sub.loc[:,'price_doc'].to_frame().to_csv('../Chase/submissions/052717_linear_ridge_regression.csv')

sub.sort_values('price_doc')

# df.loc[test_index,'nominal_index']

sub.head()




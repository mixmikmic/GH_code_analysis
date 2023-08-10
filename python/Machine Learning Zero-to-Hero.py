import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df_train = pd.read_csv("train.csv", index_col='Id')
df_test = pd.read_csv("test.csv", index_col='Id')

df_train.head()

target = df_train['SalePrice'] 
df_train = df_train.drop('SalePrice', axis=1)

df_train['training_set'] = True
df_test['training_set'] = False

df_full = pd.concat([df_train, df_test])

df_full = df_full.interpolate()
df_full = pd.get_dummies(df_full)

df_train = df_full[df_full['training_set']==True]
df_train = df_train.drop('training_set', axis=1)
df_test = df_full[df_full['training_set']==False]
df_test = df_test.drop('training_set', axis=1)

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(df_train, target)

preds = rf.predict(df_test)

my_submission = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds})
my_submission.to_csv('submission.csv', index=False)


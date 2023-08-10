import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import ensemble, metrics
from IPython.display import Image
from IPython.core.display import HTML 

# limit to first 1000 rows for now until doc is complete
hdd = pd.read_csv('../input/harddrive_resampled.csv') #,nrows = 10000)
hdd.head()

# number of rows and columns in dataset
hdd.shape

train_y = np.asarray(hdd['failure'])
if 1 in train_y:
    print ("Has failures")
else:
    print ("No failures")


hdd.drop(['failure', 'serial_number'], inplace=True, axis=1)
hdd.columns
feat_names = hdd.columns.values
feat_names



#from sklearn import ensemble
#model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
#model.fit(train_df, train_y)

## plot the importances ##
#importances = model.feature_importances_
#std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
#indices = np.argsort(importances)[::-1][:20]

#plt.figure(figsize=(12,12))
#plt.title("Feature importances")
#plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
#plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
#plt.xlim([-1, len(indices)])
#plt.show()
#'''

import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(hdd, train_y, feature_names=hdd.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()



from sklearn.ensemble import RandomForestRegressor

# try random forrest feature importance
#rf = ensemble.RandomForestClassifier()
#rf.fit(hdd, train_y)
# get rankings of feature importance
#preds = rf.predict_proba(test)
rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=1)
rf.fit(hdd, train_y)
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feat_names), reverse=True)
#ranks = ranking(rf.feature_importances_, feat_names)

# Put the mean scores into a Pandas dataframe
#meanplot = pd.DataFrame(ranks.items(), columns= ['Feature','Ranking'])

# Sort the dataframe
#meanplot = meanplot.sort_values('Ranking', ascending=False)

use_columns = ['smart_5_raw', 'smart_11_raw', 'smart_187_raw', 'smart_189_raw',                'smart_196_raw', 'smart_197_raw', 'smart_198_raw']

columns_to_drop =['smart_2_raw', 'smart_3_raw', 'smart_7_raw', 'smart_8_raw', 'smart_11_raw', 'smart_192_raw',  'smart_195_raw', 
    'smart_199_raw', 'smart_200_raw', 'smart_220_raw', 'smart_222_raw', 'smart_223_raw',  'smart_191_raw',
    'smart_224_raw',  'smart_225_raw', 'smart_226_raw', 'smart_240_raw',  'smart_241_raw', 
    'smart_242_raw',  'smart_250_raw', 'smart_251_raw',  'smart_252_raw',  'smart_254_raw',
    'smart_22_raw', 'smart_188_raw']

hdd.drop(columns_to_drop, inplace=True, axis=1)
hdd.columns

hdd.shape

# based on above need to drop some more columns -- moving up in notebook- for better graph
columns_to_drop =['smart_10_raw', 'smart_13_raw', 'smart_196_raw', 'smart_201_raw']
hdd.drop(columns_to_drop, inplace=True, axis=1)
hdd.columns

feats_df = pd.DataFrame()
feats_df = hdd
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(feats_df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


from __future__ import print_function
import sys
import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)

from util.load_data import JSONData

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.expand_frame_repr', False)

# Program Parameters
FEATURE_DIR = '../Data/train_feat/'
DATA_ROOT = '../Data/dataset/'

# Load Dataset
data = pd.read_csv(FEATURE_DIR+'feat.csv', sep=',', header=None).astype(float)

# Perform Pre-Filter to Remove Some Erraneous Records
data = data.replace([np.inf, -np.inf, np.nan], 0)

# Load Original Raw Dataset
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
train_Y = data_load.load_train_Y()

data.head()

data.describe()

# Feature NULL Type Analysis
# Looks like our preprocessing routine screwed up big time with the n-gram based features.
print(data.isnull().any().values)

# Type for Each Feature Seems to be Correct
print(data.dtypes.values)

# Drop First ID Col and Reindex Header
data = data.drop(data.columns[[0]],axis=1)
data.columns = range(len(data.columns))
data.head()

# Extract Values
Y_target = np.array(map(lambda x: 0 if x['truthClass'] == 'no-clickbait' else 1, train_Y))
Y_mean = np.array(map(lambda x: x['truthMean'], train_Y))
Y_mod = np.array(map(lambda x: x['truthMode'], train_Y))
Y_var = np.array(map(lambda x: np.var(x['truthJudgments']), train_Y))
Y_id = map(lambda x: int(x['id']), train_Y)

# Filter Out Data Frame By Index
f_data = data.iloc[Y_id, :]

# Feature to Mean Correlation
sorted(zip(range(119), map(lambda i: np.corrcoef(f_data[i].values, Y_mean)[0][1]**2, range(119))), key=lambda tup: tup[1], reverse=True)

# Feature to Variance Correlation
sorted(zip(range(119), map(lambda i: np.corrcoef(f_data[i].values, Y_var)[0][1]**2, range(119))), key=lambda tup: tup[1], reverse=True)

# Feature to Mode Correlation
sorted(zip(range(119), map(lambda i: np.corrcoef(f_data[i].values, Y_mod)[0][1]**2, range(119))), key=lambda tup: tup[1], reverse=True)

pw_corr = map(lambda x: map(lambda y: np.corrcoef(x, y), f_data.values), f_data.values)

def calculate_vif_(X, thresh=5.0):
    variables = range(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + str(X[variables].columns[maxloc]) + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[variables]

key_var = calculate_vif_(data)

f, ax = plt.subplots(figsize=(11, 11))
plt.title('Pearson Correlation of Features')
sns.heatmap(data.corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=False)
plt.xticks([])
plt.yticks([])
plt.show()

vif_idx = [  7,  16,  17,  21,  23,  24,  27,  28,  29,  30,  33,  34,  35,
             36,  38,  40,  41,  42,  43,  45,  48,  49,  50,  51,  52,  53,
             54,  55,  63,  72,  73,  76,  77,  79,  80,  81,  82,  84,  87,
             88,  89,  90,  91,  94,  95,  96,  97,  98,  99, 100, 101, 103,
            104, 107, 110]
f, ax = plt.subplots(figsize=(11, 11))
plt.title('Pearson Correlation of VIF Proposed Features')
sns.heatmap(data.iloc[:, vif_idx].corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=False)
plt.show()

lr = LinearRegression(normalize=True)
lr.fit(f_data,Y_mean)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(f_data,Y_mean)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), range(120), order=-1)

rfe_list = sorted([(k,v) for k, v in ranks['RFE'].iteritems()], key=lambda x: x[1])
rfe_list[:10]

# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(f_data, Y_mean)
ranks["LinReg"] = ranking(np.abs(lr.coef_), range(120))

# Using Ridge 
ridge = Ridge(alpha = 7)
ridge.fit(f_data, Y_mean)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), range(120))

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(f_data, Y_mean)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), range(120))

lr_rank = sorted([(k,v) for k,v in ranks['LinReg'].iteritems()], key=lambda x: x[1], reverse=True)
ridge_rank = sorted([(k,v) for k,v in ranks['Ridge'].iteritems()], key=lambda x: x[1], reverse=True)
lasso_rank = sorted([(k,v) for k,v in ranks['Lasso'].iteritems()], key=lambda x: x[1], reverse=True)

print('TOP 10 LINREG RANK')
lr_rank[:10]

print('TOP 10 LASSO RANK')
lasso_rank[:10]

print('TOP 10 RIDGE RANK')
ridge_rank[:10]

rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(f_data, Y_mean)
ranks["RF"] = ranking(rf.feature_importances_, range(120))

rf_rank = sorted([(k,v) for k,v in ranks['RF'].iteritems()], key=lambda x: x[1], reverse=True)
rf_rank

r = {}
for name in range(119):
    r[name] = round(np.mean([ranks[method][name]for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
for name in range(119):
    print("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))

mean_list = sorted([(k,v) for k, v in ranks['Mean'].iteritems()], key=lambda x: x[1], reverse=True)
top_15 = [i[0] for i in mean_list[:15]]

f, ax = plt.subplots(figsize=(11, 11))
plt.title('Pairwise Pearson Correlation of Top 15 Features')
sns.heatmap(data.iloc[:, top_15].corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)
plt.show()


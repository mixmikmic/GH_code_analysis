import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import preprocess_string,preprocess_documents
from gensim import corpora,models, similarities
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, StratifiedKFold
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
from sklearn import metrics

df1 = pd.read_csv('newdfcleaned.csv', encoding="ISO-8859-1")
df1 = df1.drop('Unnamed: 0', 1)
df1.head()

df2 = pd.read_csv('essay_features.csv')
df2.fillna(0, inplace=True)
full_df = df1.merge(df2, how = 'left', left_on='essay_id', right_on='essay_id')
full_df.head()

c_vect = CountVectorizer(stop_words='english', max_features=200, ngram_range=(1, 1))
c_vect.fit(df1['essay'])

c_vect_sparse_1 = c_vect.transform(df1['essay'])
c_vect_sparse1_cols = c_vect.get_feature_names()

pred_feats = ['essay_set'] + full_df.columns.values[6:].tolist()
df_cv1_sparse = sparse.hstack((full_df[pred_feats].astype(float), c_vect_sparse_1)).tocsr()

SEED = 777
NFOLDS = 5

params = {
    'eta':.01,
    'colsample_bytree':.8,
    'subsample':.8,
    'seed':0,
    'nthread':16,
    'objective':'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class':7,
    'silent':1
}

x_train = df_cv1_sparse.toarray()
dtrain = xgb.DMatrix(data=x_train, label=y_train1)

bst = xgb.cv(params, dtrain, 10000, NFOLDS, early_stopping_rounds=50, verbose_eval=25)

best_rounds = np.argmin(bst['test-mlogloss-mean'])

bst = xgb.train(params, dtrain, best_rounds)

dtrain = xgb.DMatrix(data=x_train)
xgb_pred = bst.predict(dtrain)
print "log loss of probability predictions: ", metrics.log_loss(y_train1, xgb_pred)
preds = pd.DataFrame(xgb_pred)

preds['essay_id'] = df1.essay_id.values

preds['scores'] = preds[[0,1,2,3,4,5,6]].idxmax(axis = 1)
metrics.accuracy_score(y_train1, preds.scores)


from __future__ import division 
# This tells matplotlib not to try opening a new window for each plot.
#%matplotlib inline

# General libraries.
import os
import codecs
import json
import csv

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import glob
import pickle
import time
# SK-learn libraries for learning.
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn library for importing the newsgroup data.
from sklearn.datasets import fetch_20newsgroups

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import NMF

from nltk.stem import WordNetLemmatizer
import nltk

get_ipython().run_cell_magic('time', '', 'data=pickle.load(open("../../Objects/Fulldata_wY_correct", \'rb\'))\ndata=data.drop(["Unnamed: 0"], axis=1)')

get_ipython().run_cell_magic('time', '', 'wordnet_lemmatizer = WordNetLemmatizer()\nlemmatized_text=[wordnet_lemmatizer.lemmatize(text) for text in data.text]')

lemmatized_text[0][0:1000]

get_ipython().run_cell_magic('time', '', "vectorizer1=CountVectorizer(stop_words='english', min_df=10)\ntext_vector=vectorizer1.fit_transform(lemmatized_text)\ntext_vector.shape")

pickle.dump(text_vector, open("../../Objects/Text_vector_lemmatized", 'wb'))

get_ipython().run_cell_magic('time', '', '##  select top features using feature selection packages\nlabels=data.label\nktop=SelectKBest(chi2, k=3000).fit_transform(text_vector, labels)\nktop.shape')

get_ipython().run_cell_magic('time', '', "## Non-negative factorization of the top unigram features, with 100 dimensions\nmodel100 = NMF(n_components=100, init='random', random_state=1, alpha=.1, l1_ratio=.5)\ntopVec100 = model100.fit_transform(ktop)\n")

alldata=pd.DataFrame(np.hstack((data.as_matrix(), topVec100)))
alldata.columns=np.array(['Company', 'ticker', 'Surprise', 'Reported_EPS', 'Consensus_EPS',
       'Date', 'timestamp', 'bow', 'items', 'text', 'orig_file',
       'release_time_type', 'return', 'stock_performance',
       'market_performance', 'normalized_performance', 'label']+range(100))
allfeatures=alldata.drop(["Company", "ticker",'bow', 'orig_file', 'stock_performance',                               'market_performance', 'normalized_performance', 'text',                          'timestamp' , 'Reported_EPS', 'Consensus_EPS', "items", "return", 'Surprise', 'release_time_type'], axis=1).dropna(axis=0, how="any")
allfeatures.head(3)

train_data = allfeatures.loc[allfeatures.Date < pd.to_datetime('2009-01-01'), :].drop(['Date', 'label'], axis=1)
dev_data = allfeatures.loc[(allfeatures.Date >= pd.to_datetime('2009-01-01')) &                            (allfeatures.Date <= pd.to_datetime('2010-12-31')), :].drop(['Date', 'label'], axis=1)
test_data = allfeatures.loc[allfeatures.Date >= pd.to_datetime('2011-01-01'), :].drop(['Date'], axis=1)
test_label=test_data['label']
test_data=test_data.drop(["label"], axis=1)

train_label=allfeatures.loc[allfeatures.Date < pd.to_datetime('2009-01-01'), "label"]
dev_label = allfeatures.loc[(allfeatures.Date >= pd.to_datetime('2009-01-01')) &                            (allfeatures.Date <= pd.to_datetime('2010-12-31')), 'label']

print train_data.shape, dev_data.shape, test_data.shape

rf=RandomForestClassifier(n_estimators=2000)
model_text=rf.fit(train_data, train_label)

get_ipython().run_cell_magic('time', '', '# Dev set accuracy\npreds_dev = model_text.predict(dev_data)\nF_Score_dev = metrics.f1_score(dev_label, preds_dev, average=\'weighted\')\n#model_output(pred_probas, F_Score, preds)\nconf_dev=confusion_matrix(dev_label.values, preds_dev,labels=["UP", "STAY", "DOWN"] , )\nprint(conf_dev/len(preds_dev))\nprint("F-score : {:3.3f}".format(F_Score_dev))\nprint("Accuracy : {:3.3f}".format(np.sum(preds_dev==dev_label)/len(dev_label)))')

# Test set accuracy
preds_test = model_text.predict(test_data)
F_Score_test = metrics.f1_score(test_label, preds_test, average='weighted')
pred_probas = model_text.predict_proba(test_data)
#model_output(pred_probas, F_Score, preds)
conf_test=confusion_matrix(test_label, preds_test)
print(conf_test/len(preds_test))
print("F-score : {:3.3f}".format(F_Score_test))
print("Accuracy : {:3.3f}".format(np.sum(preds_test==test_label)/len(test_label)))

alldata=pd.DataFrame(np.hstack((data.as_matrix(), topVec100)))
alldata.columns=np.array(['Company', 'ticker', 'Surprise', 'Reported_EPS', 'Consensus_EPS',
       'Date', 'timestamp', 'bow', 'items', 'text', 'orig_file',
       'release_time_type', 'return', 'stock_performance',
       'market_performance', 'normalized_performance', 'label']+range(100))
allfeatures=alldata.drop(["Company", "ticker",'bow', 'orig_file', 'stock_performance',                               'market_performance', 'normalized_performance', 'text',                          'timestamp' , 'Reported_EPS', 'Consensus_EPS', "items", "return"], axis=1).dropna(axis=0, how="any")
allfeatures.head()

train_data = allfeatures.loc[allfeatures.Date < pd.to_datetime('2009-01-01'), :].drop(['Date', 'label'], axis=1)
dev_data = allfeatures.loc[(allfeatures.Date >= pd.to_datetime('2009-01-01')) &                            (allfeatures.Date <= pd.to_datetime('2010-12-31')), :].drop(['Date', 'label'], axis=1)
test_data = allfeatures.loc[allfeatures.Date >= pd.to_datetime('2011-01-01'), :].drop(['Date'], axis=1)
test_label=test_data['label']
test_data=test_data.drop(["label"], axis=1)

train_label=allfeatures.loc[allfeatures.Date < pd.to_datetime('2009-01-01'), "label"]
dev_label = allfeatures.loc[(allfeatures.Date >= pd.to_datetime('2009-01-01')) &                            (allfeatures.Date <= pd.to_datetime('2010-12-31')), 'label']

print train_data.shape, dev_data.shape, test_data.shape

rf=RandomForestClassifier(n_estimators=2000)
model_lem=rf.fit(train_data, train_label)

# Dev set accuracy
preds_dev = model_lem.predict(dev_data)
F_Score_dev = metrics.f1_score(dev_label, preds, average='weighted')
pred_probas_dev = model_lem.predict_proba(dev_data)
#model_output(pred_probas, F_Score, preds)
conf_dev=confusion_matrix(dev_label.values, preds_dev,labels=["UP", "STAY", "DOWN"] , )
print(conf_dev/len(preds_dev))
print("F-score : {:3.3f}".format(F_Score_dev))
print("Accuracy : {:3.3f}".format(np.sum(preds_dev==dev_label)/len(dev_label)))

# Test set accuracy
preds_test = model.predict(test_data)
F_Score_test = metrics.f1_score(test_label, preds_test, average='weighted')
pred_probas = model.predict_proba(test_data)
#model_output(pred_probas, F_Score, preds)
conf_test=confusion_matrix(test_label, preds_test)
print(conf_test/len(preds_test))
print("F-score : {:3.3f}".format(F_Score_test))
print("Accuracy : {:3.3f}".format(np.sum(preds_test==test_label)/len(test_label)))


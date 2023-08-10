import os
import json
import time
import pickle
import requests
import math


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.DataFrame()
df = pd.read_csv('may_june_july.csv', delimiter="|")
#ab = xy[:10]
#df = ab.copy()
#df = df[df.metav != 'news']
df = df.reset_index(drop=True)
#df.tail()


# Combine all text
df['AllText'] = ""
df['primary_kw'].fillna(" ", inplace=True)
df['tags'].fillna(" ", inplace=True)
for i, row in df.iterrows():
    #cv = df.iloc[i,5]+" "+df.iloc[i,6]+" "+df.iloc[i,7]+" "+df.iloc[i,8]+" "+df.iloc[i,9]+" "+df.iloc[i,10]
    #Remove metav and cat
    cv = df.iloc[i,5]+" "+df.iloc[i,6]+" "+df.iloc[i,7]+" "+df.iloc[i,9]+" "+df.iloc[i,10]
    df.set_value(i,'AllText',cv)

# Log to convert to Normal Distribution
df['Log'] = df['freq']*df['impressions']/1000
for i, row in df.iterrows():
    cv = math.log(df.iloc[i,12],2)
    df.set_value(i,'Log',cv)
    
# analyse data a bit
data_mean = df["Log"].mean()
print data_mean
data_std = df["Log"].std()
print data_std
get_ipython().magic('matplotlib inline')
plt.hist(df["Log"])
plt.show()

# Virality defined as -1 sigma from mean
df['viral'] = np.where(df['Log']<data_mean-data_std, 'notviral', 'viral')
df['viral_num'] = df.viral.map({'notviral':0, 'viral':1})

X = df.AllText
y = df.viral_num
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# instantiate the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_df=0.1)

# learn training data vocabulary, then use it to create a document-term matrix
# FOLLOWING CAN BE DONE IN SINGLE STEP:  X_train_dtm = vect.fit_transform(X_train)
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# train the model using X_train_dtm (timing it with an IPython "magic command")
get_ipython().magic('time nb.fit(X_train_dtm, y_train)')

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)

# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# train the model using X_train_dtm
get_ipython().magic('time logreg.fit(X_train_dtm, y_train)')

# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)

# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
y_pred_prob

# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)

# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)

# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_

# rows represent classes, columns represent tokens
nb.feature_count_.shape

# number of times each token appears across all Non-viral Buzzes
non_viral_token_count = nb.feature_count_[0, :]
non_viral_token_count

# number of times each token appears across all Viral Buzzes
viral_token_count = nb.feature_count_[1, :]
viral_token_count

# create a DataFrame of tokens with their separate non-viral and viral counts
tokens = pd.DataFrame({'token':X_train_tokens, 'non_viral':non_viral_token_count, 'viral':viral_token_count}).set_index('token')
#tokens.head()

# examine 5 random DataFrame rows
#tokens.sample(20, random_state=6)

# Naive Bayes counts the number of observations in each class
nb.class_count_

# add 1 to non-viral and viral counts to avoid dividing by 0
tokens['non_viral'] = tokens.non_viral + 1
tokens['viral'] = tokens.viral + 1
#tokens.sample(5, random_state=6)

# convert the non-viral and viral counts into frequencies
tokens['non_viral'] = tokens.non_viral / nb.class_count_[0]
tokens['viral'] = tokens.viral / nb.class_count_[1]
#tokens.sample(5, random_state=6)

# calculate the ratio of viral-to-non-viral for each token
tokens['viral_ratio'] = tokens.viral / tokens.non_viral
#tokens.sample(5, random_state=6)

# examine the DataFrame sorted by viral_ratio
# note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
#tokens.sort_values('viral_ratio', ascending=False)

# look up the viral_ratio for a given token
tokens.loc['stanford', 'viral_ratio']








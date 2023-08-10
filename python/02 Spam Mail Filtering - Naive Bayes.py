import glob
import pandas as pd
import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix 
import numpy as np 
import scipy as sp 
import matplotlib as mpl 
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 
import pandas as pd 
import nltk
import re
import csv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
from time import time

spam_data = pd.DataFrame(columns = ("Text", "Class")) 

for cla in glob.glob("dataset/*"): # Here the folder name is data set in which there are two sub-folder "Spam" & "Ham"
    clas = cla.split(os.sep)[1]    # We are splitting the folder names as class using OS-Seperator and taking the 2nd item in the list
    for file in glob.glob(cla + "/*.txt"): # Here we are deep diving in each of the folder and reading the text files one by one
        text = open(file, "r", encoding = "ISO-8859-1").read() # Reading the file , for Windows generally we need to mention the encoding 
        text = " ".join(text.split("\n")) # Splitting the text files and rejoining into a single text
        spam_data = spam_data.append(pd.Series([text, clas], index = ["Text", "Class"]), ignore_index = True) # continious append to the data frame

spam_data.head()

# convert label to a numerical variable
spam_data['Class'] = spam_data.Class.map({'ham':0, 'spam':1})

stop = set(stopwords.words('english'))

spam_complete = spam_data['Text'].tolist()

len(spam_complete)

spam_complete[5]

spam_complete[7]

def clean(doc):
    doc = " ".join([i.replace('*', '') for i in doc.lower().split()])
    doc = " ".join([i.replace(':', ' ') for i in doc.split()])
    doc = " ".join([i.replace('.', ' ') for i in doc.split()])
    doc = " ".join([i.replace('=', '') for i in doc.split()])
    doc = " ".join([i.replace('/', ' ') for i in doc.split()])
    doc = " ".join([i.replace(')', ' ') for i in doc.split()])
    doc = " ".join([i.replace('(', ' ') for i in doc.split()])
    doc = " ".join([i.replace('"', ' ') for i in doc.split()])
    doc = " ".join([i.replace('-', ' ') for i in doc.split()])
    doc = " ".join([i.replace('_', ' ') for i in doc.split()])
    doc = " ".join([i for i in doc.split() if not i.isdigit()])
    doc = " ".join([i for i in doc.split() if i.isalpha()])
    doc = " ".join([i for i in doc.split() if i not in stop])
    return doc

spam_clear = [clean(doc) for doc in spam_complete]

spam_clear[7]

spam_data["clean_text"] = spam_clear

spam_data.head()

spam_data_ABT=spam_data[["Class","clean_text"]]

spam_data_ABT.head()

# how to define X and y (from the mail spam data) for use with COUNTVECTORIZER
X = spam_data_ABT.clean_text
y = spam_data_ABT.Class
print(X.shape)
print(y.shape)

# split X and y into training and testing sets
# by default, it splits 75% training and 25% test
# random_state=1 for reproducibility
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 2. instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix

# 3. fit
vect.fit(X_train)

# 4. transform training data
X_train_dtm = vect.transform(X_train)

# equivalently: combine fit and transform into a single step
# this is faster and what most people would do
X_train_dtm = vect.fit_transform(X_train)

# examine the document-term matrix
X_train_dtm

# 4. transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm

# you can see that the number of columns, 7456, is the same as what we have learned above in X_train_dtm

# 2. instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()

# 3. train the model 
# using X_train_dtm (timing it with an IPython "magic command")

get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')

# 4. make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# examine class distribution
print(y_test.value_counts())
# there is a majority class of 0 here, hence the classes are skewed

# calculate null accuracy (for multi-class classification problems)
# .head(1) assesses the value 1208
null_accuracy = y_test.value_counts().head(1) / len(y_test)
print('Null accuracy:', null_accuracy)

# Manual calculation of null accuracy by always predicting the majority class
print('Manual null accuracy:',(1208 / (1208 + 185)))

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

# print message text for the false positives (ham incorrectly classified as spam)

X_test[y_pred_class > y_test]


# print message text for the false negatives (spam incorrectly classified as ham)

X_test[y_pred_class < y_test]

# calculate AUC
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
metrics.roc_auc_score(y_test, y_pred_prob)

# 1. import
from sklearn.linear_model import LogisticRegression

# 2. instantiate a logistic regression model
logreg = LogisticRegression()

# 3. train the model using X_train_dtm
get_ipython().run_line_magic('time', 'logreg.fit(X_train_dtm, y_train)')

# 4. make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)

# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)




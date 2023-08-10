from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
import sklearn.metrics as sk

import pandas as pd
from collections import Counter
import numpy as np
import nltk

import matplotlib.pyplot as plt
import seaborn
get_ipython().magic('matplotlib inline')

modern = pd.read_pickle('data/5color_modern_no_name_hardmode.pkl')
Counter(modern.colors)

modern['bincolor'] = pd.Categorical.from_array(modern.colors).codes

vectorizer = CountVectorizer()

y = modern.bincolor

X = vectorizer.fit_transform(modern.text)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=2, multi_class='ovr', solver='liblinear')

acc = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "Accuracy: %s" % acc.mean().round(3)

n=1200
num_feat = []
acc_feat = []

for i in xrange(n/10 -1):
    vectorizer = CountVectorizer(max_features=n)

    X = vectorizer.fit_transform(modern.text)
    
    num_feat += [len(vectorizer.vocabulary_)]

    print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    clf = LogisticRegression(C=2, multi_class='ovr', solver='liblinear')

    acc = cross_val_score(clf, X_train, y_train,
                           cv=7, scoring='accuracy') 
    
    acc_feat += [acc.mean()]

    print "Accuracy: %s" % acc.mean().round(3)
    
    n -= 10
    
    

num_feat

acc_feat #= [i.mean() for i in acc_feat]

# seaborn.set_style("darkgrid")
plt.plot(zip(num_feat, acc_feat))
plt.show()






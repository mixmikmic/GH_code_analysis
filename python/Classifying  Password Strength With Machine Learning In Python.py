# EDA Packages
import pandas as pd
import numpy as np
import random


# Machine Learning Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Passwordlist
# Credits Faizan
#pswd_data = pd.read_csv("passwordsdata.csv",error_bad_lines=False)
pswd_data = pd.read_csv("cleanpasswordlist.csv")

pswd_data.head()

pswd = np.array(pswd_data)



# Shuffle data 
random.shuffle(pswd)

# Grouping our data into features and labels
# y = labels
ylabels  = [s[2] for s in pswd]
allpasswords = [s[1] for s in pswd]

ylabels
len(ylabels)

len(allpasswords)

def makeTokens(f):
    tokens = []
    for i in f:
        tokens.append(i)
    return tokens

# Using Default Tokenizer
#vectorizer = TfidfVectorizer()

# Using Custom Tokenizer
vectorizer = TfidfVectorizer(tokenizer=makeTokens)

# Store vectors into X variable as Our X allpasswords
X = vectorizer.fit_transform(allpasswords)

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)	



# Model Building
#using logistic regression
# Multi_class for fast algorithm
logit = LogisticRegression(penalty='l2',multi_class='ovr')	
logit.fit(X_train, y_train)

print("Accuracy :",logit.score(X_test, y_test))

X_predict = ['password',
             'pYthon'
             'faizanahmad',
             'password##',
             'ajd1348#28t**',
             'ffffffffff',
             'kuiqwasdi',
             '123456',
             'abcdef']

X_predict = vectorizer.transform(X_predict)
y_Predict = logit.predict(X_predict)
print(y_Predict)

# http://www.passwordmeter.com/
# https://password.kaspersky.com/

New_predict = ["1qaz2wsx",
"306187mn",
"rados1#@1$#@$@$#@$",
"newyork911",
"abc123",
"taqiyudin100587",
"wjr5443",
"nana0428"]

New_predict = vectorizer.transform(New_predict)
y_Predict = logit.predict(New_predict)
print(y_Predict)



# Using Default Tokenizer
vectorizer = TfidfVectorizer()
# Store vectors into X variable as Our X allpasswords
X1 = vectorizer.fit_transform(allpasswords)

X_train, X_test, y_train, y_test = train_test_split(X1, ylabels, test_size=0.2, random_state=42)	

# Model Building
#using logistic regression
# Multi_class for fast algorithm
logit = LogisticRegression(penalty='l2',multi_class='ovr')	
logit.fit(X_train, y_train)

print("Accuracy :",logit.score(X_test, y_test))

# Using th

X_predict1 = ['password',
             'pYthonqwas'
             'faizanahmad',
             'password##',
             'ajd1348#28t**',
             'ffffffffff',
             'kuiqwasdi',
             '123456',
             'abcdef']

X_predict1 = vectorizer.transform(X_predict1)
y_Predict1 = logit.predict(X_predict1)
print(y_Predict1)

New_predict2 = ["1qaz2wsx",
"306187mn",
"rados1",
"newyork911",
"abc123",
"taqiyudin100587",
"wjr5443",
"nana0428"]

New_predict2 = vectorizer.transform(New_predict2)
y_Predict2 = logit.predict(New_predict2)
print(y_Predict2)

#Thanks 
#J-Secur1ty
#Jesus Saves @ JCharisTech


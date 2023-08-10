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
#df = df[df.pull_cc == 'us']
#df = df.reset_index(drop=True)
df.head()



# Combine all text
df['AllText'] = ""
df['primary_kw'].fillna(" ", inplace=True)
df['tags'].fillna(" ", inplace=True)
for i, row in df.iterrows():
    #cv = df.iloc[i,5]+" "+df.iloc[i,6]+" "+df.iloc[i,7]+" "+df.iloc[i,8]+" "+df.iloc[i,9]+" "+df.iloc[i,10]
    #Remove metav and cat
    cv = df.iloc[i,2]+" "+df.iloc[i,5]+" "+df.iloc[i,6]+" "+df.iloc[i,7]+" "+df.iloc[i,9]+" "+df.iloc[i,10]
    df.set_value(i,'AllText',cv)

print df.tail()


# Log to convert to Normal Distribution
df['Log'] = df['freq']*(df['impressions']+1)/1000

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

# Assign buzzes
df['viral'] = ""
for i, row in df.iterrows():
    if df.iloc[i,12]<=(data_mean-1.5*data_std):
        df.set_value(i,'viral','1buzz')
    elif (df.iloc[i,12]>(data_mean+1.5*data_std)):
        df.set_value(i,'viral','3buzz')
    else:
        df.set_value(i,'viral','2buzz')


#df['viral'] = np.where(df['Log']<data_mean-1*data_std, 'notviral', 'viral')
df['viral_num'] = 0
df['viral_num'] = df.viral.map({'1buzz':1, '2buzz':2, '3buzz':3})

X = df.AllText
y = df.viral_num
# instantiate the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_df=0.1)
df.head()

df.tail()

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

from sklearn.pipeline import make_pipeline
pipe=make_pipeline(vect, nb)
pipe.steps

# calculate accuracy of class predictions
from sklearn.cross_validation import cross_val_score
cross_val_score(pipe,X,y,cv=12,scoring='accuracy').mean()

# import and instantiate a Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.pipeline import make_pipeline
pipe=make_pipeline(vect, logreg)
pipe.steps

# calculate accuracy of class predictions
cross_val_score(pipe,X,y,cv=12,scoring='accuracy').mean()

data_mean-1.5*data_std

data_mean+1.5*data_std

print data_mean
print data_std

df.shape

df.viral.value_counts()




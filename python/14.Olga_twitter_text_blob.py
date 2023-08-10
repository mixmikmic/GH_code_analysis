import nltk

import re
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

import csv

import os
import matplotlib.pyplot as plt


import sys
import os
import time

import random


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

print os.getcwd();

os.chdir("/Users/imacair/Desktop/Products3/")

data = pd.read_csv('Final_Manual_0805.csv', encoding= "latin-1",delimiter=',',low_memory=False)
#data


data.head(10)

#Converts text into ASCII

data.message = data.message.str.encode('ascii','replace')

data.sentiment = data.sentiment.str.encode('utf-8','replace')

#number of elements
len(data)

#In case if you need a sample of data
#sample_data= data.sample(n=64000)

sample_data= data

data_t=sample_data["message"]

len(data_t)

#lowercase before abbriviation translation
data_t = data_t.str.lower()

data_s=sample_data["sentiment"]



with open('abbrev.csv', mode='r') as infile:
    reader = csv.reader(infile)
    replacement = {rows[0].lower():rows[1].lower() for rows in reader              
                  }

#replacement

#replacement = {
##'r':'are',
#'y':'why',
#'u':'you'}

##How in works
s1 = 'y r u l8'

s2 = ' '.join([replacement.get(w, w) for w in s1.split()])
s2

result = pd.DataFrame()
result = data_t

for i in range(len(result)):
    data_t.values[i]=' '.join([replacement.get(w, w) for w in data_t.values[i].split()])

data_t.head(10)

#lowercase
data_t = data_t.str.lower()
#Remove urls
data_t= data_t.str.replace(r'(http.*) |(http.*)$|\n', "",)
#Remove twitter handles
data_t = data_t.str.replace(r"@\\w+", "")
#remove htmls
data_t = data_t.str.replace(r'<.*?>', "")
#Remove citations
data_t = data_t.str.replace(r'@[a-zA-Z0-9]*', "")

#remove _
#data_t = data_t.str.replace(r'\_+',"")

data_t.head(10)

from textblob import TextBlob

data_t.head(10)

#Creating a column for polarity
res= pd.DataFrame(np.random.randn(len(data_t), 1),columns = {'polarity'} )

#Creating a column for subjectivity
res2= pd.DataFrame(np.random.randn(len(data_t), 1),columns = {'subjectivity'} )

for i in range(len(data_t)):
    testimonial = TextBlob(data_t.values[i])
    res.values[i]= testimonial.sentiment.polarity

for i in range(len(data_t)):
    testimonial = TextBlob(data_t.values[i])
    res2.values[i]= testimonial.sentiment.subjectivity

final=pd.concat([data_t, res], axis=1)

final=pd.concat([final, res2], axis=1)

final=pd.concat([final, data_s], axis=1)

#Test on one value
testimonial = TextBlob(data_t.values[2])
testimonial.sentiment.polarity

final

res6= pd.DataFrame( index=range(0,len(data_t)),columns = {'SentimentPat'} )

#Creating a clumn with clear sentiment
res6.SentimentPat[(final['polarity']>0)]='pos'
res6.SentimentPat[(final['polarity']<0)]='neg'
res6.SentimentPat[(final['polarity']==0)]='neu'

final=pd.concat([final, res6], axis=1)

from sklearn.metrics import confusion_matrix

#final.sentiment[which(final.sentiment=="1")]<-"neg"

final

from textblob.sentiments import NaiveBayesAnalyzer

from textblob import Blobber
tb = Blobber(analyzer=NaiveBayesAnalyzer())

print tb("sentence you want to test").sentiment.classification

tb(data_t[3]).sentiment.p_pos

#Creating a column for results
res4= pd.DataFrame( index=range(0,len(data_t)),columns = {'sentimentNB','pposNB','pnegNB'} )

res4

for i in range(len(data_t)):
    res4['sentimentNB'][i]= tb(data_t[i]).sentiment.classification
    res4['pposNB'].values[i]= tb(data_t[i]).sentiment.p_pos
    res4['pnegNB'].values[i]= tb(data_t[i]).sentiment.p_neg

res4

final=pd.concat([final, res4], axis=2)

final

final.sentimentNB[(final['pposNB']<0.7) &(final['pposNB']>0.3) & (final['pnegNB']<0.7) & (final['pnegNB']>0.3)]='neu'

res5= pd.DataFrame( index=range(0,len(data_t)),columns = {'new_sent'} )
#final.sentiment[1][(final['sentiment']==u'2') &(final['sentiment']==u'1')] = 'neg'
#final=pd.concat([final, res5], axis=2)

final.sentiment[0]

res5[(final.sentiment==u'2')] = 'neg'
res5[(final.sentiment==u'1')] = 'neg'
res5[(final['sentiment']==u'3')] = 'pos'
res5[(final['sentiment']==u'4')] = 'pos'
res5[(final['sentiment']==u'N')] = 'neu'
final=pd.concat([final, res5], axis=2)
#res5

final

from sklearn.metrics import confusion_matrix

print(confusion_matrix(final.new_sent, final.sentimentNB).transpose())

print(confusion_matrix(final.new_sent, final.SentimentPat).transpose())

print(len(final[(final['sentimentNB']=='neg') & (final['new_sent']=='pos')]))
##print(len(final[(final['sentimentNB']=='pos') & (final['sentiment']<u'3')]))
#print(len(final[(final['sentimentNB']=='neg') & (final['sentiment']>u'2')]))
#print(len(final[(final['sentimentNB']=='neg') & (final['sentiment']<u'3')])) 


#Download the libraries
import nltk
import re
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import sys
import time
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from sklearn.externals import joblib
import cPickle as pickle

print os.getcwd();

os.chdir("/Users/imacair/Desktop/Products3/")

data_t= "I love Humira"

#Converts text into ASCII

#data.text = data.text.str.encode('ascii','replace')

#data.message = data.message.str.encode('ascii','replace')

len(data)

#sample_data= data.sample(n=64000)



sample_data= data

#data_t=sample_data["text"]

#data_t=sample_data["message"]

len(data_t)



#lowercase
data_t = data_t.lower()

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
result = final

for i in range(len(result)):
    data_t.values[i]=' '.join([replacement.get(w, w) for w in data_t.values[i].split()])


text = data_t
text = nltk.word_tokenize(text)
fdist = nltk.FreqDist(text)
s2 = set([w for w in set(text) if len(w) > 2 and fdist[w] > 2])
for i in range(len(result)):
    data_t.values[i]=' '.join(filter(lambda w: w in s2,data_t.values[i].split()))



from nltk.corpus import stopwords
s=set(stopwords.words('english'))
for i in range(len(result)):
    data_t.values[i]=' '.join(filter(lambda w: not w in s,data_t.values[i].split()))

data_t

#lowercase
data_t = data_t.lower()
#Remove urls
data_t= data_t.replace(r'(http.*) |(http.*)$|\n', "",)
#Remove twitter handles
data_t = data_t.replace(r"@\\w+", "")
#remove htmls
data_t = data_t.replace(r'<.*?>', "")
#Remove citations
data_t = data_t.replace(r'@[a-zA-Z0-9]*', "")
#remove _
#data_t = data_t.str.replace(r'\_+',"")

data_t



#Use vader package to get the sentiment
analyzer = SentimentIntensityAnalyzer()
res= pd.DataFrame( index=range(0,1),columns = {'SentimentVader'} )

#Convert sentiment to neu, neg, pos
for i in range(1):
    vs = analyzer.polarity_scores(data_t)
    if ((vs['pos']>0)):
        res.values[i]= 'pos' 
    elif ((vs['neg'] < 0)):
        res.values[i]= 'neg'
    else:
        res.values[i]= 'neu'
vader = res.SentimentVader

vader

#Use textblob to get polarity of text
res6= data_t
testimonial = TextBlob(data_t)
res6= testimonial.sentiment.polarity
#Convert polarity to normal pos, neg, neu
textblob1= res6
if ((res6>0)):
    textblob1= 'pos' 
elif ((res6<0)):
    textblob1= 'neg' 
else:
    textblob1= 'neu' 

textblob1

data_t

#Use textblob to get polarity of text with Naive Bayes analyzer
tb = Blobber(analyzer=NaiveBayesAnalyzer())
textblob2= pd.DataFrame( index=range(0,1),columns = {'sentimentNB'} )
textblob2['sentimentNB']= tb(data_t).sentiment.classification

textblob2

vader = np.asarray(vader)

#Create a Data Frame
df=[vader]
df = pd.DataFrame(df)
df = df.transpose()
df.columns = [ 'vader']
df["SentimentPat"] = textblob1
df["sentimentNB"] = textblob2
df

#Find the maximum in a row (Majority voting)
df2= pd.DataFrame( index=range(0,1),columns = {'final'} )
for i in range(1):
    d=Counter(df.ix[i,:])
    dfin=d.most_common(1)[0][0]
    df2.values[i]= dfin
df["final"] = df2

df2




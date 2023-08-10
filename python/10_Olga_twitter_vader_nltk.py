import sys
import os
import time
import csv

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report



print os.getcwd();

os.chdir("/Users/imacair/Desktop/Products3/")

data = pd.read_csv('Final_Manual_0805.csv', encoding= "latin-1",delimiter=',',low_memory=False)

#Converts text into ASCII
data.message = data.message.str.encode('ascii','replace')
data.sentiment = data.sentiment.str.encode('utf-8','replace')

len(data)

data_t=data["message"]

#lowercase before abbriviation translation
data_t = data_t.str.lower()

#takes sentiment
data_s=data["sentiment"]

with open('abbrev.csv', mode='r') as infile:
    reader = csv.reader(infile)
    replacement = {rows[0].lower():rows[1].lower() for rows in reader              
                  }

##How in works
s1 = 'y r u l8'

s2 = ' '.join([replacement.get(w, w) for w in s1.split()])
s2

result = pd.DataFrame()
result = data_t

for i in range(len(result)):
    data_t.values[i]=' '.join([replacement.get(w, w) for w in data_t.values[i].split()])

data_t.head()

type(result)

data_t=result

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

#Start of sentiment analysis

analyzer = SentimentIntensityAnalyzer()

analyzer.polarity_scores(data_t[190])

res= pd.DataFrame( index=range(0,len(data_t)),columns = {'SentimentVader'} )

for i in range(len(data_t)):
    vs = analyzer.polarity_scores(data_t.values[i])
    if ((vs['compound']==0)):
        res.values[i]= 'neu' 
    elif ((vs['compound'] < 0)):
        res.values[i]= 'neg'
    else:
        res.values[i]= 'pos'
    

res.SentimentVader

data_t.head(10)

from textblob import TextBlob

final=pd.concat([data_t, res], axis=1)
final=pd.concat([final, data_s], axis=1)

res5= pd.DataFrame( index=range(0,len(data_t)),columns = {'new_sent'} )
res5[(final.sentiment==u'2')] = 'neg'
res5[(final.sentiment==u'1')] = 'neg'
res5[(final['sentiment']==u'3')] = 'pos'
res5[(final['sentiment']==u'4')] = 'pos'
res5[(final['sentiment']==u'N')] = 'neu'
final=pd.concat([final, res5], axis=2)

final

from sklearn.metrics import confusion_matrix

print(confusion_matrix(final.new_sent, final.SentimentVader).transpose())




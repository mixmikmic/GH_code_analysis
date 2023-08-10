get_ipython().magic('matplotlib inline')
import csv
import pandas
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold, cross_val_score 

# Read the data
reviews = [line.rstrip() for line in open("/Users/mam/CORE/RESEARCH/DEEPLEARNING/Doc2Vec/data/aclImdb/alldata_2column.txt")]
print(len(reviews))

# The data have a header and we print it
print(reviews[0])
# print first data point.
# data format is each review as a line, csv
# clomun one is the sentiment tag --> 1=positive sentiment, 0=negative sentiment
# column 2 is the review
print(reviews[1])

# Let's actually read the file again with pandas
import csv
import pandas as pd
reviews = pd.read_csv("/Users/mam/CORE/RESEARCH/DEEPLEARNING/Doc2Vec/data/aclImdb/alldata_2column.txt",                      sep=',', quoting=csv.QUOTE_NONE,  names=["label", "message"])

# Let's print a preview with the "head" command
reviews.head(n=5)

reviews_data=reviews["message"]
reviews_tags=reviews["label"]

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='word')),  # get counts of tokens
    ('tfidf', TfidfTransformer()),  # get tf-idf scores
    ('classifier', MultinomialNB()),  # train on tf-idf vectors  with the Naive Bayes classifier
])

# Do 10-fold cross validation
scores = cross_val_score(pipeline,  
                         reviews_data,  
                         reviews_tags,  
                         cv=10, 
                         scoring='accuracy',
                         n_jobs=-1, # use all machine cores
                         )
print(scores)

# Let's get average accuracy...
avg= sum(scores/10.0)
print(avg)


# Download data
# NLTK Corpora Twitter Samples
# http://www.nltk.org/nltk_data/
import requests, zipfile, io

url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/twitter_samples.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

import os
import json
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle


curr_dir = os.getcwd()

classes = ['pos', 'neg']

_data = []
_labels = []
for line in open (os.path.join(curr_dir ,"twitter_samples/") + r'positive_tweets.json', 'r'):
    _data.append(json.loads(line)['text'])
    _labels.append("pos")


for line in open (os.path.join(curr_dir ,"twitter_samples/") + r'negative_tweets.json', 'r'):
    _data.append(json.loads(line)['text'])
    _labels.append("neg")

#### Create feature vectors
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8, stop_words='english', use_idf=True)

data_vectors = vectorizer.fit_transform(_data)

cv=KFold(data_vectors.shape[0], n_folds=10, shuffle=True, random_state=1)

# Perform classification with MultinomialNB
clf = MultinomialNB()

scores = cross_validation.cross_val_score(clf, data_vectors, _labels, cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
clf.fit(data_vectors, _labels)

with open('models/clf.pkl', 'wb') as fmodel:
    pickle.dump(clf, fmodel)
with open('models/vocabulary.pkl', 'wb') as fvocabulary:
    pickle.dump(vectorizer.vocabulary_, fvocabulary)

with open('models/clf.pkl', 'rb') as fmodel:
    clf = pickle.load(fmodel)
with open('models/vocabulary.pkl', 'rb') as fvocabulary:
    vocabulary = pickle.load(fvocabulary)

import tweepy
from tweepy import OAuthHandler
 
consumer_key = 'yourkey'
consumer_secret = 'yourkey'
access_token = 'yourkey'
access_secret = 'yourkey'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

with open('tweets.csv', 'w') as csvfile:
    tweet_writer = csv.writer(csvfile)
    for tweet in tweepy.Cursor(api.search, q='trump', languages=["en"]).items(50):
        tweet_writer.writerow([tweet.text.encode('utf-8')])

tweets = []
with open('tweets.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        tweets.append(row[0])

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8, stop_words='english', use_idf=True, vocabulary=vocabulary)
tweet_vectors = vectorizer.fit_transform(tweets)

pred_multinomial = clf.predict(tweet_vectors)
prob_multinomial = clf.predict_proba(tweet_vectors)

sentiment_tweets = []
for index, prob in enumerate(prob_multinomial):
    sentiment_tweets.append({"Tweet": tweets[index], "p_neg": prob[0], "p_pos": prob[1], "target": pred_multinomial[index]})

with open('tweet_sentiments.json', 'w') as jsonfile:
    json.dump(sentiment_tweets, jsonfile, sort_keys=True, indent=4, ensure_ascii=True)

sentiment_tweets[:5]




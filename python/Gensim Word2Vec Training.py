from gensim.models.word2vec import Word2Vec
from gensim import corpora, models, similarities
import gensim, logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import numpy as np
import csv
import sklearn as sc
import os
import json
from bs4 import BeautifulSoup
import re

get_ipython().magic('matplotlib inline')



data_dir = r'C:/Users/csa77/Documents/Data'
home_dir = r'C:/Users/csa77/Documents/GitHub/Twitter_graphing_python/streaming_results'

# Read data from twitter sentiment data file

train = {}

with open(os.path.join(data_dir, 'Sentiment Analysis Dataset.csv'), encoding="utf-8") as f:
    
    reader = csv.reader(f)
    next(reader)
    
    for i, row in enumerate(reader):
        identity, sentiment, source, text, *extra = row
        categories = 'positive negative'.split()
        train[i] = {'target':sentiment, 'data':text, 'words': [], 'target_names': categories[int(sentiment)]}
    
    

data = pd.DataFrame.from_dict(train, orient='index')

data.head()

def tweet_to_wordlist(tweet, remove_stopwords=False):
    #tweet_text = BeautifulSoup(tweet).get_text()
    #tweet_text = re.sub("[^a-zA-Z]", " ", tweet)
    words = tweet.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.tweet("english"))
        words = [w for wo in words if not w in stops]
        
    return words

data.words = [tweet_to_wordlist(tweet) for tweet in data.data]

for t in data[:10]:
    print(data.target_names)

tweets = data.words
tweets[:10]

targets = np.array(data.target)
targets[:10]

categories = "negative positive".split()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(data.data)
vectors.shape

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    data.data, targets, test_size=0.4)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)
X_train_counts.shape

count_vect.vocabulary_.get(u'algorithm')

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_tfidf = tfidf_transformer.fit(X_train_counts)
X_train__tfidf = tfidf_transformer.transform(X_train_counts)
X_train__tfidf.shape

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB().fit(X_train__tfidf, y_train)

docs_new = ["Superman is so cool.", "I'm not sure if it was worthwhile",
           "It was pretty sucky.", "I'll never talk to him again."]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('{} => {}'.format(doc, categories[int(category)]))

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    # Simple plot based on the Iris sample CM
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
sgd_tweet_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                          alpha=1e-3, n_iter=5,
                                          random_state=42))
                     ])

_ = sgd_tweet_clf.fit(x_train, y_train)

sgd_predicted = sgd_tweet_clf.predict(x_test)
np.mean(sgd_predicted == y_test)

print(metrics.classification_report(y_test, sgd_predicted, target_names=categories))

sgd_cm = metrics.confusion_matrix(y_test, sgd_predicted)

plot_confusion_matrix(sgd_cm, categories, title="SGD Confusion Matrix")

from sklearn.naive_bayes import MultinomialNB
naive_bayes_tweet_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())
                     ])

_ = naive_bayes_tweet_clf.fit(x_train, y_train)

naive_bayes_predicted = naive_bayes_tweet_clf.predict(x_test)
np.mean(naive_bayes_predicted == y_test)

naive_bayes_cm = metrics.confusion_matrix(y_test, naive_bayes_predicted)

print(naive_bayes_cm)

print(metrics.classification_report(y_test, naive_bayes_predicted, target_names=categories))

plot_confusion_matrix(naive_bayes_cm, categories, title='Naive Bayes')

def load_tweets(target):
    # Output list of tweets
    tweets = []

    with open(os.path.join(home_dir, target)) as f:
        for data in f:

            result = json.loads(data)
            
            try:
                tweets.append(result['text'])
            except KeyError:
                continue
    
    return tweets

def process_tweets(tweets):
    # Output processed list of tweets

    texts = [[word for word in tweet.lower().split()] for tweet in tweets]
    
    # Remove words that only occur once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    from pprint import pprint
    #pprint(texts)
    
    return texts

# Download the harper tweest
harper_tweets = load_tweets('harper_stream.json')

# Run raw tweets through Pipeline
harper_predicted = naive_bayes_tweet_clf.predict(harper_tweets)

# Convert result predicted values to integers
harper_predicted = [int(z) for z in harper_predicted]

# Print raw tweet and predicted value
for doc, category in zip(harper_tweets, harper_predicted):
    print('{} => {}\n'.format(doc, categories[int(category)]))

print("Average {}".format(sum(harper_predicted) / len(harper_predicted)))

trudeau_tweets = load_tweets('trudeau_stream.json')
trudeau_predicted = naive_bayes_tweet_clf.predict(trudeau_tweets)

trudeau_predicted = [int(z) for z in trudeau_predicted]

print("Average {}".format(sum(trudeau_predicted) / len(trudeau_predicted)))

new_predicts = naive_bayes_tweet_clf.predict(docs_new)

for doc, category in zip(docs_new, new_predicts):
    print('{} => {}\n'.format(doc, categories[int(category)]))

trial = ["Bloodborne follows the player character, the Hunter, through the fictional decrepit Gothic city of Yharnam, whose inhabitants have been afflicted with an abnormal blood-borne disease. Upon mysteriously awakening in Yharnam during the night of 'The Hunt', the Hunter seeks out something known only as 'Paleblood' for reasons unknown.[5] The Hunter begins to unravel Yharnam's intriguing mysteries while hunting down its many terrifying beasts. Eventually, the Hunter's objective is to locate and terminate the source of the plague, and escape the nightmare to return to the 'real world', known as the 'Waking World'."]

trial2 = ["The Great Depression was a severe worldwide economic depression in the 1930s. The timing of the Great Depression varied across nations; however, in most countries it started in 1929 and lasted until the late 1930s.[1] It was the longest, deepest, and most widespread depression of the 20th century.[2] Worldwide GDP fell by 15% from 1929 to 1932.[3] In the 21st century, the Great Depression is commonly used as an example of how far the world's economy can decline.[4] The depression originated in the United States, after the fall in stock prices that began around September 4, 1929, and became worldwide news with the stock market crash of October 29, 1929 (known as Black Tuesday)."]

trial_predicts = naive_bayes_tweet_clf.predict(trial)

trial2_predicts = naive_bayes_tweet_clf.predict(trial2)

print('{} => {}'.format(trial, categories[int(trial_predicts[0])]))
print("")
print('{} => {}'.format(trial2, categories[int(trial2_predicts[0])]))



processed_harper_tweets = process_tweets(harper_tweets)
dictionary = corpora.Dictionary(processed_harper_tweets)

harper_vec = [dictionary.doc2bow(tweet) for tweet in processed_harper_tweets]

harper_vec

trial3 = "Never have I seen a more foul man conduct himself so well."

trial3_predicts = naive_bayes_tweet_clf.predict(trial3)

print('{} => {}'.format(trial3, categories[int(trial3_predicts[0])]))




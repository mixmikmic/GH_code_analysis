import numpy as np
import pandas as pd
import sqlite3
import gensim
import re
from nltk.corpus import stopwords
import nltk

# Use the NLTK downloader to download stopwords and punkt tokenizer (for breaking paragraphs into sentences)

#nltk.download()

get_ipython().system('ls ../data')

sql_conn = sqlite3.connect('../data/database.sqlite')

mathematics = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'mathematics'",sql_conn)

computerscience = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'computerscience'",sql_conn)

history = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'history'",sql_conn)

philosophy = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'philosophy'",sql_conn)

elifive = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'explainlikeimfive'",sql_conn)

askanthro = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'AskAnthropology'",sql_conn)

homebrewing = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'Homebrewing'",sql_conn)

bicycling = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'bicycling'", sql_conn)

food = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'food'", sql_conn)

science = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'science'", sql_conn)

movies = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'movies'", sql_conn)

books = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'books'", sql_conn)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')

# Array of tuples, with df and subject
subreddits = [(bicycling,'bicycling'),(history,'history'),(philosophy,'philosophy'),
              (elifive,'explain'),(homebrewing,'homebrew'),(askanthro,'anthropology'),
              (mathematics,'mathematics'),(computerscience,'computer science'),
              (food,'food'),(science,'science'),(movies,'movies'),(books,'books')]

for (subreddit,subject) in subreddits:
    print(subject+'\n')
    print(subreddit.info())
    print('=========\n')

all_frames = [bicycling, history, philosophy, elifive, homebrewing, askanthro, mathematics,              computerscience, food, science, movies, books]
model_training_data = pd.concat(all_frames, ignore_index=True)

model_training_data.info()

# Takes a sentence in a comment and converts it to a list of words.
def comment_to_wordlist(comment, remove_stopwords=False ):
    comment = re.sub("[^a-zA-Z]"," ", comment)
    words = comment.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

def comment_to_sentence(comment, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(comment.strip())
    
    sentences = []
    for s in raw_sentences:
        if len(s)>0:
            sentences.append(comment_to_wordlist(s, remove_stopwords))
    #rof
    return sentences

# Download a tokenizer to parse comments into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

comment_to_sentence(model_training_data.loc[1,'body'], tokenizer)

sentences = []
for comment in model_training_data['body']:
    sentences += comment_to_sentence(comment, tokenizer)

print(len(sentences))

num_features = 300
min_word_count = [10, 30, 50, 100]
context = [3, 5, 10]
downsampling = 1e-5
num_workers = 4

from gensim.models import word2vec
import time
import logging
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

num = 300

for mwc in min_word_count:
    for c in context:
        # Initialize and train each model 
        print("Training model with the following parameters:")
        message = "Size of feature vector: {}\nMin word count: {}\nn-gram context: {}\nDownsampling of frequent words: {}\n".format(num, mwc, c, downsampling)
        print(message)
        
        start = time.time();
        model = word2vec.Word2Vec(sentences, workers=num_workers,                     size=num, min_count = mwc,                     window = c, sample = downsampling)
        end = time.time();
        total = end-start;
        # Compress, name, and store each model
        model.init_sims(replace=True)
        model_name = str(num) + "features_" + str(mwc) + "minwords_" + str(c) + "context"
        print("Took " + str(total) + " time to train model: " + model_name)
        model.save(model_name)
        print("Model saved.")

# This cell is used for training an individual model
from gensim.models import word2vec
import time
import logging

# Initialize and train each model 
print("Training model with the following parameters:")
message = "Size of feature vector: {}\nMin word count: {}\nn-gram context: {}\nDownsampling of frequent words: {}\n".format(300, 10, 3, downsampling)
print(message)
        
start = time.time();
model = word2vec.Word2Vec(sentences, workers=num_workers,             size=300, min_count = 10,             window = 10, sample = downsampling)
end = time.time();
total = end-start;
# Compress, name, and store each model
model.init_sims(replace=True)
model_name = str(300) + "features_" + str(10) + "minwords_" + str(10) + "context"
print("Took " + str(total) + " time to train model: " + model_name)
model.save(model_name)
print("Model saved.")

model.doesnt_match("man woman child kitchen".split())

sum((model["man"]-model["child"])**2)

# Not sure if this model is that great, I'm thinking the context window was too large
# Model 1:
#model.init_sims(replace=True)
#model_name = "300features_30minwords_15context_includescience"
#model.save(model_name)

# Can load the model later with Word2Vec.load()

training_frames = [bike_training, hist_training, phil_training, elif_training, brew_training]
training_data = pd.concat(training_frames, ignore_index=True)

training_data.info()

training_sentences = []
for comment in training_data['body']:
    training_sentences += comment_to_sentence(comment, tokenizer, True)

# Filter out sentences with less than 5 words, these are likely nonsensical
after_filter_training_sentences = filter(lambda x: len(x)>5, training_sentences)

print(len(training_sentences))
print(len(after_filter_training_sentences))

labels = ["bicycle","statistics","history","philosophy","homebrewing","anthropology","explain"]

def f(word,label):
    return abs(model.similarity(word, label))

def label(sentence,labels):
    # Initialize distance to be high
    best_distance = 1e8
    l = ""
    for label in labels:
        ssds = map(lambda x: f(x,label),sentence)
        #print("Label: " + label)
        #print(ssds)
        #average = sum(ssds)/len(ssds)
        if min(ssds) < best_distance:
            l = label
            best_distance = min(ssds)
    return l

test = after_filter_training_sentences[1]
print(test)
label(test,labels)


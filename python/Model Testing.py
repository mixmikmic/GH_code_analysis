import numpy as np
import pandas as pd
import sqlite3
import gensim
import re
from nltk.corpus import stopwords
import nltk
import matplotlib
from gensim.models import word2vec

sql_conn = sqlite3.connect('../data/database.sqlite')

# These functions are needed for processing later

# Takes a sentence in a comment and converts it to a list of words.
def comment_to_wordlist(comment, remove_stopwords=False ):
    comment = re.sub("[^a-zA-Z]"," ", comment)
    words = comment.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

# Takes a comment and converts it to an array of sentences
def comment_to_sentence(comment, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(comment.strip())
    
    sentences = []
    for s in raw_sentences:
        if len(s)>0:
            sentences.append(comment_to_wordlist(s, remove_stopwords))
    #rof
    return sentences

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

# Array of tuples, with df and subject
subreddits = [(bicycling,'bicycling'),(history,'history'),(philosophy,'philosophy'),
              (elifive,'explain'),(homebrewing,'homebrew'),(askanthro,'anthropology'),
              (mathematics,'mathematics'),(computerscience,'computer'),
              (food,"food"),(science,"science"),(movies,"movies"),(books,"books")]

all_frames = [bicycling, history, philosophy, elifive, homebrewing, askanthro, mathematics,              computerscience, food, science, movies, books]
model_training_data = pd.concat(all_frames, ignore_index=True)
model_training_data.info()

# Get all model files
from os import listdir
from os.path import isfile, join
models = [f for f in listdir('new_models/') if isfile(join('new_models/', f)) and not f.endswith('.npy')]
print(models)

import math

# Method for computing similarity to a specific label
# Note this returns the cosine similarity between 
# the vector representation of the words and label
def similarities(comment, label, model):
    dists = []
    for word in comment:
        if word in model.vocab:
            dists.append(model.similarity(word,label))
    return dists

def label_comment(comment, labels, model):
    # Set initial distance to be 1-(-1) (complete dissimilarity)
    best_distance = 2
    best_label = ""
    for label in labels:
        # Range will be from [-1,1]
        word_dists_to_label = similarities(comment, label, model)
        
        # We want to choose the label with overall sum closest to 1
        # or
        # We want to minimize our distance to 1
        dist = 1 - sum(word_dists_to_label)
        if dist < best_distance:
            best_label = label
            best_distance = dist
    return best_label

from gensim.models import word2vec

# Vocab size

# Attempt to load each model
# Display size of vocabulary for each model
# As expected, lowering the minimum number of appearances of a word increases
# the size of the vocabulary

xs = []
size_of_vocab=[]

for m in models:
    print(m)
    current_model = word2vec.Word2Vec.load('new_models/' + m);
    print("Size of vocab: " + str(current_model.syn0.shape[0]))
    
    x = m.split("_")[1]
    xs.append(x.split("min")[0])
    size_of_vocab.append(current_model.syn0.shape[0])
    
    # Take each comment, compare 

m = "300features_10minwords_10context"
current_model = word2vec.Word2Vec.load('new_models/' + m);

current_model.most_similar("mathematics")

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

comments = []
ground_truth_labels = []
for i in range(len(model_training_data)):
    comments.append(comment_to_wordlist(model_training_data.iloc[i]['body'], tokenizer))
    ground_truth_labels.append(model_training_data.iloc[i]['subreddit']) # Could also just slice this array out

if len(comments)==len(ground_truth_labels):
    print("Length is the same: ",len(comments))

#labels = np.unique(model_training_data['subreddit'])
#subreddits = [(bicycling,'bicycling'),(history,'history'),(philosophy,'philosophy'),
#              (elifive,'explain'),(homebrewing,'homebrew'),(askanthro,'anthropology'),
#              (mathematics,'mathematics'),(computerscience,'computer science'),
#              (food,"food"),(gaming,"gaming"),(politics,"politics")]

subreddit_labels = []
for frame,name in subreddits:
    subreddit_labels.append(name)
    
trained_labels = []

import time

#trained_labels = [None] * len(comments)
start = time.time()
for i in range(100000):
    trained_labels.append(label_comment(comments[i],subreddit_labels,current_model))
end = time.time()
duration = end-start

print("Took " + str(duration) + " time to label 100000 comments.")

len(trained_labels)

mis_rate = 0
successes = []
for i in range(len(trained_labels)):
    if trained_labels[i] != ground_truth_labels[i]:
        mis_rate += 1
    else:
        successes.append((comments[i],trained_labels[i]))
error_rate = mis_rate/float(len(trained_labels))
success_rate = 1-error_rate

len(successes)


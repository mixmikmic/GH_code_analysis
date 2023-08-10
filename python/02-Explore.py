from __future__ import (absolute_import,
                        division)

import pandas as pd
from collections import Counter
import numpy as np
import nltk

import matplotlib.pyplot as plt
import seaborn
get_ipython().magic('matplotlib inline')

# import modern magic cards from file 
modern = pd.read_pickle('data/5color_modern_no_name_hardmode.pkl')

modern.head(2)

# drop nans and get a basic scatter matrix 

numvals = modern[['cmc', 'power', 'toughness']]
numvals = numvals[numvals['cmc'].notnull()]
numvals = numvals[numvals['power'].notnull()]
numvals = numvals[numvals['toughness'].notnull()]

pd.scatter_matrix(numvals);

# check class balance 

Counter(modern['colors'])

def countwords(x):
    return len(x)

modern['word_count'] = modern['text'].apply(countwords)

# word count by color 

modern.word_count.hist(by=modern.colors, figsize=(10,10), bins=30);

modern[modern['word_count'] >= 400][['name', 'colors', 'text']]

# resource cost by color 

modern.cmc.hist(by=modern.colors, figsize=(10,10), bins=10);

# highest resource cost cards

modern[modern['cmc'] >= 11][['name', 'colors', 'cmc']]

# slice to just creatures and display creature power by color 

modern2 = modern[modern['power'].notnull()]
modern2 = modern2[modern2['toughness'].notnull()]

modern2.power = pd.to_numeric(modern2.power, errors='coerce')

modern2.power.hist(by=modern2.colors, figsize=(10,10), bins=14)
plt.show();
    

# creature toughness by color

modern2.toughness = pd.to_numeric(modern2.toughness, errors='coerce')

modern2.toughness.hist(by=modern2.colors, figsize=(10,10), bins=14)
plt.show();

# most common card descriptions 

freq = nltk.FreqDist(modern.text)
train_common = freq.most_common()
train_common[:10]

# vocab size

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(lowercase=True)
vectorized_data = vectorizer.fit_transform(modern.text) 
vocab = len(vectorizer.vocabulary_)

print "There are %s unique words in the vocabulary." % vocab

# top 10 words 

def top_words(corpus, num, stop=None):
    vectorizer = CountVectorizer(stop_words=stop, lowercase=True)
    vectorized_data = vectorizer.fit_transform(corpus.text) 
    freqs = [(word, vectorized_data.getcol(idx).sum()) for word, idx 
             in vectorizer.vocabulary_.items()]
    words =  sorted (freqs, key = lambda x: -x[1])
    return [i[0] for i in words[:num]]

print "Top words from all cards"
top_words(modern, 10)

# top 10 words of each color 

def top_colors(sliced, num, stop=None):
    label = ["Black", "Blue", "Green", "Red", "White"]
    for l in label:
        print "Top %s words" % l
        print top_words(sliced.groupby('colors').get_group(l), num, stop)
        print 

top_colors(modern, 10)

print "Top words from all cards"
print top_words(modern, 10, stop='english')

top_colors(modern, 10, stop='english')

# with formatting 

def top_colors(sliced, num, stop=None):
    label = ["Black", "Blue", "Green", "Red", "White"]
    df = pd.DataFrame(np.repeat(" ", (num * 5)).reshape((num, 5)), columns=label)
    for l in label:
        top = top_words(sliced.groupby('colors').get_group(l), num, stop)
        for n in xrange(num):
            df[l][n] = top[n]
    return df

top_colors(modern, 10, stop='english')


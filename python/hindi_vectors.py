from __future__ import absolute_import, division, print_function

import codecs
import glob
import multiprocessing
import os
import pprint
import re

import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

get_ipython().magic('pylab inline')

nltk.download("punkt")

hindi_filenames = sorted(glob.glob("../data/hin_corp_unicode/*txt"))
#hindi_filenames

corpus_raw = u""
for file_name in hindi_filenames:
    print("Reading '{0}'...".format(file_name))
    with codecs.open(file_name, "r", "utf-8") as f:
        # Starting two lines are not useful in corpus
        temp = f.readline()
        temp = f.readline()
        corpus_raw += f.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)

def sentence_to_wordlist(raw):
    clean = re.sub("[.\r\n]"," ", raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

token_count = sum([len(sentence) for sentence in sentences])
print("The Hindi corpus contains {0:,} tokens".format(token_count))

sentences[0]

# Dimensionality of the resulting word vectors.
# More dimensions = more generalized
num_features = 50
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
num_threads = multiprocessing.cpu_count()

# Context window length.
context_size = 8

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
# Random Number Generator
seed = 1

# Defining the model
model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_threads,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

model.build_vocab(sentences)

model.train(sentences)

# Save our model
model.save(os.path.join("../data/", "hindi_word2Vec_small.w2v"))

trained_model = w2v.Word2Vec.load(os.path.join("../data/", "hindi_word2Vec_small.w2v"))

# For reducing dimensiomns, to visualize vectors
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = trained_model.syn1neg[:200] # Currently giving memory error for all words
# Reduced dimensions
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[trained_model.wv.vocab[word].index])
            for word in trained_model.wv.vocab
            if trained_model.wv.vocab[word].index < 200
        ]
    ],
    columns=["word", "x", "y"]
)

s = trained_model.wv[u"आधार"]




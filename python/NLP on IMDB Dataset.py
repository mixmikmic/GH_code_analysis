# read in data
import pandas as pd
train = pd.read_csv('data/labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
print(train["review"][0])
print(train["sentiment"][0])

# cleaning data
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re
def clean(review):
    # remove html
    text = BeautifulSoup(review, "html5lib").get_text()
    # regexp matching to extract letters only
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    # remove common words
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))

cleaned_reviews = [clean(train["review"][i]) for i in range(len(train["review"]))]

print(cleaned_reviews[1].split())

from collections import defaultdict
import numpy as np
# creates a vocabulary - set of all words in all reviews
def create_vocab(cleaned_reviews):
    """
    Takes in a bunch of reviews and creates a vocabulary. 
    """
    li = []
    for review in cleaned_reviews:
        a = review.split()
        for item in a:
            li.append(item)
    return list(set(li))

def get_word_occ_dict(review):
    d = defaultdict(int)
    words = review.split()
    for w in words:
        d[w]+=1
    return d

# takes in a vocab and a review and returns a feature vector for the review
# the feature vector f has d dimensions where d = len (vocab)
# for the i in [1..d]th word, f[i] = n where n is the number of times the word occured in the review
# the feature vectors are sparse, since most words in the vocab may not occur in a specific review
def create_feature_vector(review, vocab):
    word_dict = get_word_occ_dict(review) 
    feature_vector = [word_dict[v] if v in word_dict else 0 for v in vocab]
    return np.array(feature_vector)

def create_feature_vectors(cleaned_reviews, vocab):
    feature_vectors = [create_feature_vector(review, vocab) for review in cleaned_reviews]
    return np.array(feature_vectors)

vocab = create_vocab(cleaned_reviews)
X = create_feature_vectors(cleaned_reviews, vocab)

y = train['sentiment']
X.shape
y.shape

from utils import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# separate data into training and testing
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.3)


# fit SVM to the data
clf = LinearSVC(verbose = 10)
clf.fit(X_train, y_train)
y_train_pred, y_test_pred = clf.predict(X_train), clf.predict(X_test)
from sklearn.metrics import accuracy_score
test_acc, train_acc = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
print("test accuracy: {}".format(test_acc))
print("training accuracy: {}".format(train_acc))

# try several different classifiers by changing the value for C, which indicates how much slack variables are penalized.
clfs_and_params = [(LinearSVC(C = c, verbose = 10), c) for c in [0.01, 0.1, 1.0, 5.0, 10, 100]]
clf, best_params, best_test_err, best_train_err = get_best_hyperparams_cv(X_train, y_train, k = 10, 
                                                                          classifiers = clfs_and_params, 
                                                                          verbose = True)

test_data = pd.read_csv('data/unlabeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
cleaned_reviews = [clean(test_data["review"][i]) for i in range(len(test_data["review"]))]
X = create_feature_vectors(cleaned_reviews, vocab)
final_test_preds = clf.predict(X)

import tensorflow as tf

# cleaned reviews are a bunch of reviews where we will get our training examples from.
# Let's look at one cleaned review: 
print(cleaned_reviews[0])
window_size = 1
vocab = create_vocab(cleaned_reviews)

def word_one_hot(word, vocab):
    idx = vocab.index(word)
    if idx < 0:
        return -1
    vec = np.zeros((len(vocab)))
    vec[idx] = 1
    return vec

def create_vectorized_word_pairs(review, vocab, window_size):
    words = review.split()
    data = []
    for i in range(len(words)):
        left = [words[i-j] for j in range(1, window_size + 1) if i-j >= 0]
        right = [words[i+j] for j in range(1, window_size + 1) if i+j < len(words)]
        neighbors = left + right
        pairs = [(word_one_hot(words[i], vocab), word_one_hot(n,vocab)) for n in neighbors]
        data.append(pairs)
    
    return data

def create_word_pairs_all_reviews(cleaned_reviews, vocab, window_size):
    data = []
    for review in cleaned_reviews:
        li = create_vectorized_word_pairs(review, vocab, window_size)
        data = data + li
    return data

example_pairs = create_vectorized_word_pairs(cleaned_reviews[0], vocab, 1)
print(example_pairs[0])
print(cleaned_reviews[0])

# example_pairs is a list of lists where each list contains 2 * window_size elements. 
# each element will be a pair of (example, label)
# map data into concrete X/Y input output lists

features, labels = [], []
#features = [elm[0] for elm in li for lin in example_pairs]
#labels = [elm[1] for elm in li for li in example_pairs]
for li in example_pairs:
    features = features + [elm[0] for elm in li]
    labels = labels + [elm[1] for elm in li]

# TODO - implement model - probably will be written up in a module and imported here. 




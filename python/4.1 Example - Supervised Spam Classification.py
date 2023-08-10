get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

X = ["It was a bright cold day in April, and the clocks were striking thirteen",
    "The sky above the port was the color of television, tuned to a dead channel"]

len(X)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X)

vectorizer.vocabulary_

X_bag_of_words = vectorizer.transform(X)

X_bag_of_words

X_bag_of_words.shape

X_bag_of_words.toarray()

vectorizer.get_feature_names()

vectorizer.inverse_transform(X_bag_of_words)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X)

import numpy as np
np.set_printoptions(precision=2)

print(tfidf_vectorizer.transform(X).toarray())

# look at sequences of tokens of minimum length 2 and maximum length 2
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigram_vectorizer.fit(X)

bigram_vectorizer.get_feature_names()

bigram_vectorizer.transform(X).toarray()

gram_vectorizer = CountVectorizer(ngram_range=(1, 2))
gram_vectorizer.fit(X)

gram_vectorizer.get_feature_names()

gram_vectorizer.transform(X).toarray()

char_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer="char")
char_vectorizer.fit(X)

print(char_vectorizer.get_feature_names())

import os
with open(os.path.join("data","SMSSpamCollection")) as f:
    lines = [line.strip().split("\t") for line in f.readlines()]
text = [x[1] for x in lines]
y = [x[0] == "ham" for x in lines]

from sklearn.cross_validation import train_test_split

text_train, text_test, y_train, y_test = train_test_split(text, y, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

print(len(vectorizer.vocabulary_))

print(vectorizer.get_feature_names()[:20])

print(vectorizer.get_feature_names()[3000:3020])

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier()
clf

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

clf.score(X_train, y_train)




get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)

X = ["Some say the world will end in fire,",
     "Some say in ice."]

len(X)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X)
vectorizer.vocabulary_

X_bag_of_words = vectorizer.transform(X)
X_bag_of_words

print(X_bag_of_words.toarray())

print(X)
vectorizer.get_feature_names()

vectorizer.inverse_transform(X_bag_of_words)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X)

print(tfidf_vectorizer.get_feature_names())
print(tfidf_vectorizer.transform(X).toarray())

X

# look at sequences of tokens of minimum length 2 and maximum length 2
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigram_vectorizer.fit(X)
bigram_vectorizer.get_feature_names()

bigram_vectorizer.transform(X).toarray()

gram_vectorizer = CountVectorizer(ngram_range=(1, 2))
gram_vectorizer.fit(X)

gram_vectorizer.get_feature_names()

X_1_2_gram = gram_vectorizer.transform(X)
print(X_1_2_gram.shape)
print(X_1_2_gram.toarray())

char_vectorizer = CountVectorizer(ngram_range=(2, 3), analyzer="char")
char_vectorizer.fit(X)
print(char_vectorizer.get_feature_names())


import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

text = "i am paradox. i am gru. i am nish. i am a caffeine addict.  i love caffeine"

# use simple counts as features
from sklearn.feature_extraction.text import CountVectorizer

# let's train the feature extractor on this simple training "text
tokens = nltk.word_tokenize(text)
count_vectorizer = CountVectorizer().fit(tokens)
count_vectorizer

# all the available features/tokens
count_vectorizer.get_feature_names()

# let's test new document
test = "caffeine is love"
test_vect = count_vectorizer.transform(nltk.word_tokenize(test))
print(test_vect)

# get document-term matrix
test_vect.toarray()

# see what document-term matrix is under the hood
pd.DataFrame(test_vect.toarray(), columns = count_vectorizer.get_feature_names())

train_docs = [
    "Yes, it's hard to get things done, to accept stuff despite being seemingly unworthy.But hey! Worthiness is just our own abstraction of comfort.",
    "We should be able to embrace what life throws at us diligently. Being pedantic won't do good.",
    "As we are always governed by the vastness of entropy, as such we tend to be over-dramatic towards the minor things in life.",
    "But, if we can pass that out, whether the withering wealth, health, love and shit, we can probably render ourselves joyous.",
    "I think that's the way of living of life. Live. Don't just breathe."
]

test_docs = [
    "we seem to be living our life. but we are not"
]
print(train_docs)
print(test_docs)

count_vectorizer = CountVectorizer().fit(train_docs)
print(count_vectorizer.get_feature_names())

vect = count_vectorizer.transform(test_docs)
print(vect.toarray())

pd.DataFrame(vect.toarray(), columns=count_vectorizer.get_feature_names() )

# let's create tf-idf features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer().fit(train_docs)
print(tfidf_vectorizer.get_feature_names())

vect = tfidf_vectorizer.transform(test_docs)
print(vect.toarray())

pd.DataFrame(vect.toarray(), columns=tfidf_vectorizer.get_feature_names() )




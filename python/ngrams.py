import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

a = 'text about stuff'
b = 'stuff about text'
c = 'text about ngrams'
d = 'n-grams are handy'
document = [a, b, c, d]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(document)

# Creating a dictionary from term to frequency
terms = vectorizer.get_feature_names()
freqs = X.sum(axis=0).A1
result = dict(zip(terms, freqs))

result

matrix_terms = np.array(vectorizer.get_feature_names())

# Using the axis keyword to sum over rows
matrix_freq = np.asarray(X.sum(axis=0)).ravel()
final_matrix = np.array([matrix_terms,matrix_freq])

print(final_matrix)

from nltk import bigrams
sentence = 'this is a foo bar sentences and i want to ngramize it'

grams = bigrams(sentence.split())
for gram in grams:
    print(gram)


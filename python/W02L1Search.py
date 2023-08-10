import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')
stop[:5]

get_ipython().run_line_magic('matplotlib', 'inline')
nltk.download('gutenberg')
import nltk
import collections
import matplotlib.pyplot as plt
words = nltk.corpus.gutenberg.words('austen-emma.txt')
fd = collections.Counter(words)
data = sorted([fd[k] for k in fd],reverse=True)
plt.plot(data[:1000])
plt.title("Zipf's Law")
plt.xlabel("Word rank")
plt.ylabel("Frequency")

import numpy as np
a = np.array([1,2,3,4])
a[0]

a[1:3]

a+1

b = np.array([2,3,4,5])
a+b

np.dot(a,b)

x = np.array([[1,2,3],[4,5,6]])
x

y = np.array([[1,1,1],[2,2,2]])
x+y

x*y

x.T

np.dot(x.T,y)

from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile

import os.path
if not os.path.exists('enron1'):
    with zipfile.ZipFile('enron1.zip') as myzip:
        myzip.extractall()

import glob
files = glob.glob('enron1/ham/*.txt')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(input='filename',stop_words='english')
tfidf_values = tfidf.fit_transform(files)
len(tfidf.get_feature_names())

tfidf.get_feature_names()[10000:10005]

type(tfidf_values)

type(tfidf_values.toarray())

tfidf_values.shape

len(files)

import numpy as np
tfidf_norm = TfidfVectorizer(input='filename',
                             stop_words='english',
                             norm='l2')
tfidf_norm_values = tfidf_norm.fit_transform(files).toarray()
def cosine_similarity(X,Y):
    return np.dot(X,Y)

cosine_similarity(tfidf_norm_values[0,:],
                  tfidf_norm_values[1,:])

from sklearn.metrics import pairwise
pairwise.cosine_similarity([tfidf_norm_values[0,:]],
                           [tfidf_norm_values[1,:]])

pairwise.cosine_similarity([tfidf_values[0,:]],
                           [tfidf_values[1,:]])

dense_tfidf_values = tfidf_values.toarray()
pairwise.cosine_similarity([dense_tfidf_values[0,:]],
                           [dense_tfidf_values[1,:]])

pairwise.cosine_similarity([tfidf_norm_values[0,:]],
                           [tfidf_norm_values[1,:]])




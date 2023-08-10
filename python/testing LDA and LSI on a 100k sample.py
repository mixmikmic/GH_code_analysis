from pymongo import MongoClient
import re
import html.parser

from __future__ import print_function

# gensim
from gensim import corpora, models, similarities, matutils
from gensim.models import LdaMulticore
# sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import words
import nltk
import re
# logging for gensim (set to INFO)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import numpy as np
import matplotlib.pyplot as plt 

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')

client = MongoClient()
db = client.sample_100k

sample = db.sample_100k.find({
    "body": { "$exists": True },
    "$expr": { "$gt": [ { "$strLenCP": "$body"}, 300 ]} 
                                    },{
    "body": 1})

sample = list(sample)

len(sample)

sample = [x['body'] for x in sample]

sample = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', string) for string in sample]

sample = [string.replace('\n', ' ') for string in sample]

sample[4]

sample = [html.unescape(string) for string in sample]

sample[4]

# Create a CountVectorizer for parsing/counting words
count_vectorizer = CountVectorizer(ngram_range=(1, 2),  
                                   stop_words='english', 
                                   token_pattern="\\b[a-z][a-z]+\\b")
count_vectorizer.fit(sample)

counts = count_vectorizer.transform(sample).transpose()

corpus = matutils.Sparse2Corpus(counts)

id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())

lda = LdaMulticore(workers=3, corpus=corpus, num_topics=5, id2word=id2word, passes=5)

lda.print_topics(num_words=20)

lda_corpus = lda[corpus]
lda_corpus

lda_docs = [doc for doc in lda_corpus]

lda_docs[55:60]

sample[55]

sample[59]

# Vectorize the text using TFIDF
tfidf = TfidfVectorizer(stop_words="english", 
                        token_pattern="\\b[a-zA-Z][a-zA-Z]+\\b", #words with >= 2 alpha chars 
                        min_df=10,
                        ngram_range=(1, 4))
tfidf_vecs = tfidf.fit_transform(sample)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(tfidf_vecs, 
                                                    sample, 
                                                    test_size=0.33)

# Train 
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Test 
nb.score(X_test, y_test)

# Convert sparse matrix of counts to a gensim corpus
# Need to transpose it for gensim which wants 
# terms by docs instead of docs by terms
tfidf_corpus = matutils.Sparse2Corpus(tfidf_vecs.transpose())

# Row indices
id2word = dict((v, k) for k, v in tfidf.vocabulary_.items())

# This is a hack for Python 3!
id2word = corpora.Dictionary.from_corpus(tfidf_corpus, 
                                         id2word=id2word)

# Build an LSI space from the input TFIDF matrix, mapping of row id to word, and num_topics
# num_topics is the number of dimensions to reduce to after the SVD
# Analagous to "fit" in sklearn, it primes an LSI space
lsi = models.LsiModel(tfidf_corpus, id2word=id2word, num_topics=300)

# Retrieve vectors for the original tfidf corpus in the LSI space ("transform" in sklearn)
lsi_corpus = lsi[tfidf_corpus]

# Dump the resulting document vectors into a list so we can take a look
doc_vecs = [doc for doc in lsi_corpus]
doc_vecs[0]

# Create an index transformer that calculates similarity based on 
# our space
index = similarities.MatrixSimilarity(doc_vecs, 
                                      num_features=300)

# Return the sorted list of cosine similarities to the first document
sims = sorted(enumerate(index[doc_vecs[0]]), key=lambda item: -item[1])
sims

# Let's take a look at how we did
for sim_doc_id, sim_score in sims[0:3]: 
    print("Score: " + str(sim_score))
    print("Document: " + sample[sim_doc_id])

#Normalize the corpus
example = matutils.corpus2dense(lsi_corpus, num_terms=300).transpose()
example = normalize(example)

#Look at 2-20 clusters

SSEs = []
Sil_coefs = []
for k in range(2,20):
    km = KMeans(n_clusters=k, random_state=54)
    km.fit(example)
    labels = km.labels_
    Sil_coefs.append(silhouette_score(example, labels, metric='euclidean'))
    SSEs.append(km.inertia_) 

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5), sharex=True, dpi=200)
k_clusters = range(2,20)
ax1.plot(k_clusters, Sil_coefs)
ax1.set_xlabel('number of clusters')
ax1.set_ylabel('silhouette coefficient')

# plot here on ax2
ax2.plot(k_clusters, SSEs)
ax2.set_xlabel('number of clusters')
ax2.set_ylabel('SSE');

from sklearn.metrics import silhouette_samples, silhouette_score

for k in range(2,20):
    plt.figure(dpi=120, figsize=(8,6))
    ax1 = plt.gca()
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(example)
    labels = km.labels_
    silhouette_avg = silhouette_score(example, labels)
    print("For n_clusters =", k,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(example, labels)
    y_lower = 100
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[labels == i]
        
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("SILHOUETTE PLOT FOR DIFFERENT CLUSTERS")
    ax1.set_xlabel("SILHOETTE COEFFICIENT VALUES")
    ax1.set_ylabel("NUMBER OF CLUSTERS")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


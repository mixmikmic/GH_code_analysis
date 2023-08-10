import os
import classicdb.fetch as fetch
from nltk.corpus import CategorizedPlaintextCorpusReader

url = fetch.URL
data_home = fetch.get_data_home()
classic_home = os.path.join(data_home, fetch.CLASSIC_HOME)
print("download and extracting file from " + url)
fetch.download_and_unzip(url, classic_home, fetch.ARCHIVE_NAME)
print("downloaded and extracted to " + classic_home)

corpus_root = os.path.join(classic_home, fetch.TRAIN_FOLDER)
corpus_reader = CategorizedPlaintextCorpusReader(corpus_root, r'.*', cat_pattern=r'(\w+)/*')
print("database was loaded into memory")

cats = corpus_reader.categories()
print("database categories: " + str(cats))

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

labels = [cat for cat in corpus_reader.categories() for fileid in corpus_reader.fileids(cat)]
files = [corpus_reader.raw(fileid) for fileid in corpus_reader.fileids()]

start = time()
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(files)
print("done in %fs" % (time() - start))
print("m samples: %d, n features: %d" % X.shape)
print()

# terms = vectorizer.get_feature_names()
# print("some feature terms:")
# print(terms[0:100])
# print()

from sklearn.cluster import KMeans
from sklearn import metrics

km = KMeans(n_clusters=len(corpus_reader.categories()),
            init='k-means++',  # or 'random' (random centroids) 
            n_init=10,  # number of time the k-means algorithm will be run with different centroid seeds.    
            max_iter=300
            )

print("Document clustering with %s" % km)
start = time()
km.fit(X)
print("done in %0.3fs" % (time() - start))
print()

print("true labels vs cluster labels")
print(labels[0:50])
print(km.labels_[0:50])
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))
print()

get_ipython().magic('matplotlib inline')

from numpy import arange
from matplotlib import pyplot
from random import randint

def bar_chart(categories, words, counts, ylabel, title, colors='rgbcmyk', legendloc='upper left'):
    ind = arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pyplot.bar(ind + c * width, counts[categories[c]], width, color=colors[c % len(colors)])
        bar_groups.append(bars)
    pyplot.xticks(ind + width, words)
    pyplot.legend([b[0] for b in bar_groups], categories, loc=legendloc)
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    pyplot.show()

clusters = np.unique(km.labels_)
labels = [cat for cat in corpus_reader.categories() for fileid in corpus_reader.fileids(cat)]

counts = {}
for c in corpus_reader.categories():
    counts[c] = len(clusters) * [0]
for l, label in enumerate(km.labels_):
    counts[labels[l]][label] += 1
# print(counts)
bar_chart(corpus_reader.categories(), clusters, counts, "Frequency", "Composition of k-mean cluster")

counts = {}
for i, c in enumerate(clusters[0:50]):
    counts[c] = len(corpus_reader.categories()) * [0]
label_ind = {}
cnt = 0
for cat in corpus_reader.categories():
    label_ind[cat] = cnt
    cnt +=1
for l, label in enumerate(labels): 
    counts[km.labels_[l]][label_ind[label]] += 1
# print(counts)
bar_chart(clusters, corpus_reader.categories(), counts, "Frequency", "Composition of category", colors='mykw',legendloc='upper right')




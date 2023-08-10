import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import *
from sklearn.cluster import KMeans
import os
from scipy.spatial.distance import cdist, pdist
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

documents = []
names = []
docsDir = "/Users/jamesledoux/Documents/data_exploration/author_files"
for book in os.listdir(docsDir):
    if not book.startswith('.'):    #pass hidden files such as .DS_STORE
        book = str(book) #file name
        names.append(book)
        with open("/Users/jamesledoux/Documents/data_exploration/author_files/" + book, 'rb') as f:
            content = f.read() #.splitlines()
            content = unicode(content, errors='replace')
            documents.append(content)
print str(len(documents)) + " documents loaded"

stemmer = PorterStemmer()
token_docs = [nltk.word_tokenize(document) for document in documents]
token_docs = [[stemmer.stem(token) for token in doc] for doc in token_docs] #now stemmed. slow but worth it
token_docs  = [" ".join(i) for i in token_docs]
print "documents tokenized and stemmed"

tfidf = TfidfVectorizer(ngram_range=(1, 1),
                        stop_words='english',
                        strip_accents='unicode', analyzer = 'word',
                        max_features=1500,
                        min_df=2)  #should we ease up on this max df?

tfidf_matrix =  tfidf.fit_transform(token_docs)
feature_names = tfidf.get_feature_names() 
print "tfidf calculated"

K = range(1,20) #k-means k values to test
KM = [KMeans(n_clusters=k).fit(tfidf_matrix) for k in K] #fit k-means model for each k in K
centroids = [k.cluster_centers_ for k in KM]
print "k-means models complete"

X = tfidf_matrix.todense()
distances = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(d,axis=1) for d in distances]
dist = [np.min(d,axis=1) for d in distances]
avgWithinSS = [sum(d)/X.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X)**2)/X.shape[0]
bss = tss-wcss

#elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')

from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) 
ax = dendrogram(linkage_matrix, orientation="right", labels=names);

plt.tick_params(    axis= 'x',         
    which='both',      
    bottom='off',      
    top='off',        
    labelbottom='off')

plt.tight_layout()

data = pd.read_csv("features.csv")

#drop unneeded or unhelpful columns
names = data['book_name']
drops = ["book_name", "total_words", "Author", "Title", "ID", "%", "$", "&", "/", "<", ">", "+", "#", "@", "]", "~", "^", "{", "LS"]
for i in drops:
    data = data.drop(i, 1)

#aggregate similar part-of-speech features
data['verbs'] = data['VB'] + data['VBD'] + data['VBG'] + data['VBP']
data['nouns'] = data['NN'] + data['NNP'] + data['NNPS'] # ADD 'NNS' BACK IN ONCE IT'S IN THE SCRIPT
data['adjectives'] = data['JJ'] + data['JJR'] + data['JJS']

columns = data.columns
df = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(df)
data = pd.DataFrame(data_scaled)
data.columns = columns

#viewing distributions
cols = data.columns
plt.figure(figsize=(20, 16))

#histograms of all features to understand distributions
for i in range(len(cols)):
    plt.subplot(8,8,i+1)
    lab = cols[i]
    plt.hist(data[lab])
    plt.yscale('linear')
    plt.title(lab)
    plt.grid(True)
    plt.subplots_adjust(hspace=.5)
plt.show()

#correaltions between features
sns.set(context="paper", font="monospace")

corrmat = data.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 12))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)

# Use matplotlib directly to emphasize known networks
networks = corrmat.columns.get_level_values(0)

for i, network in enumerate(networks):
    if i and network != networks[i - 1]:
        ax.axhline(len(networks) - i, c="w")
        ax.axvline(i, c="w")

f.tight_layout()

#features to focus on. CD = "cardinal" (aka numbers). RB = adverbs. FW = foreign word. IN = preposition/conjunction.
pos_viz = ['verbs', 'nouns', 'CD', 'adjectives', 'RB', 'FW', 'IN']
punct_viz = ['!', "''", ':', ',', "?", ';', '(', '.', '-']

#plot results for these four authors
desired_names = ['rowling', 'rrmartin', 'einstein', 'newton']

for i in range(len(names)):
    if names[i] in desired_names:
        data_viz1 = data[pos_viz][i:i+1] #get one observation with just these columns
        data_viz2 = data[punct_viz][i:i+1] 
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        plt.suptitle(names[i], fontsize=20)
        data_viz1.plot(ax=axes[0], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')
        data_viz2.plot(ax=axes[1], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')

#drop the aggregate features since they're already represented in the individual variables
verbs, adjectives, nouns = data['verbs'], data['adjectives'], data['nouns']
data = data.drop("verbs", 1)
data = data.drop("adjectives", 1)
data = data.drop("nouns", 1)

K = range(1,15) #k-means k values to test
KM = [KMeans(n_clusters=k).fit(data) for k in K] #fit k-means model for each k in K
centroids = [k.cluster_centers_ for k in KM]

distances = [cdist(data, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(d,axis=1) for d in distances]
dist = [np.min(d,axis=1) for d in distances]
avgWithinSS = [sum(d)/data.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(data)**2)/data.shape[0]
bss = tss-wcss

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')

names = np.array(names)
labels = KM[4].predict(data)

zero = []
one = []
two = []
three = []
four = []
five = []
for i in range(len(names)):
    if labels[i] == 0:
        zero.append(names[i])
    if labels[i] == 1:
        one.append(names[i])
    if labels[i] == 2:
        two.append(names[i])
    if labels[i] == 3:
        three.append(names[i])
    if labels[i] == 4:
        four.append(names[i])
    if labels[i] == 5:
        five.append(names[i])

print "ZERO: "
print zero

print "ONE: "
print one

print "TWO: "
print two

print "THREE: "
print three

print "FOUR: "
print four

_centroids = centroids[5]
_centroids.shape
_centroids = pd.DataFrame(_centroids)
columns = data.columns

_centroids.columns = columns

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(20,18))
_centroids.iloc[0].plot(ax=axes[0], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')
_centroids.iloc[1].plot(ax=axes[1], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')
_centroids.iloc[2].plot(ax=axes[2], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')
_centroids.iloc[3].plot(ax=axes[3], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')
_centroids.iloc[4].plot(ax=axes[4], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')

from nltk.corpus import stopwords
import string
from nltk.util import bigrams
from itertools import chain

stoplist = stopwords.words('english')
stoplist = stoplist + ['e', 'u', 'n', 'h', 'i', 'l']  #hawking uses math, and the notation throws off this analysis
doc = documents[28].lower()

#remove punctuation and stopwords, tokenize
exclude = set(string.punctuation)
doc = ''.join(ch for ch in doc if ch not in exclude)
doc = nltk.word_tokenize(doc) #tokenize document
doc = [token for token in doc if token not in stoplist] #remove all stopwords

#plot top words
frequencyDistribution = nltk.FreqDist(doc)
plt.figure(figsize=(12, 6))
# plot the top 20 tokens
frequencyDistribution.plot(20)

#plot top word pairs (bi-grams)
b = list(chain(*[(list(bigrams(doc)))]))
fdist = nltk.FreqDist(b)
plt.figure(figsize=(12, 6))
# plot the top 20 bigrams
fdist.plot(20)

print "similar authors: "
print one

#generate plot of stylistic features
#data = pd.read_csv("features.csv")
pos_viz = ['verbs', 'nouns', 'CD', 'adjectives', 'RB', 'FW', 'IN']
punct_viz = ['!', "''", ':', ',', "?", ';', '(', '.', '-']
data['verbs'] = verbs
data['nouns'] = nouns 
data['adjectives'] = adjectives
desired_name = ['hawking']

for i in range(len(names)):
    if names[i] in desired_name:
        data_viz1 = data[pos_viz][i:i+1] #get one observation with just these columns
        data_viz2 = data[punct_viz][i:i+1] 
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        plt.suptitle(names[i], fontsize=20)
        data_viz1.plot(ax=axes[0], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')
        data_viz2.plot(ax=axes[1], kind='bar').set_ylim(0,1); plt.axhline(0, color='k')




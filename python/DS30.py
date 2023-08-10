# Importing graphics related libraries
get_ipython().magic('matplotlib inline')
import matplotlib
import seaborn as sns  # plots are prettier with Seaborn
import onlineldavb
from wordcloud import WordCloud
from IPython.display import Image
from IPython import display
matplotlib.rcParams['savefig.dpi'] = 2 * matplotlib.rcParams['savefig.dpi']

# importing useful libraries
import simplejson  # more efficient than the default json library
import sys
import requests  # better than the urllib libraries
from requests_oauthlib import OAuth1
from collections import Counter
import heapq
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans
from itertools import islice, chain
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pylab as plt

with open("twitter_secrets.json.nogit") as fh:
    secrets = simplejson.loads(fh.read())

auth = OAuth1(
    secrets["api_key"],
    secrets["api_secret"],
    secrets["access_token"],
    secrets["access_token_secret"]
)

US_BOUNDING_BOX = "-125.00,24.94,-66.93,49.59"
DAYTON_BOX= "-84.21,39.72,-84.045,39.80"
def tweet_generator():
    """ Generator that live streams tweets (see 'yield' keyword)"""
    stream = requests.post('https://stream.twitter.com/1.1/statuses/filter.json',
                         auth=auth,
                         stream=True,
                         data={"locations" : DAYTON_BOX})
    
    for line in stream.iter_lines():
        if not line:  # filter out keep-alive new lines
            continue
        tweet = simplejson.loads(line)        
        if 'text' in tweet:
            yield tweet



for tweet in tweet_generator():
    print tweet
    break

DISPLAY_EVERY = 20

stop = set(stopwords.words('english'))  # predefined list of "uninteresting" words

counter = Counter()

def nlargest(n, word_scores):
    """ Wrapper around heapq to return the n words with the largest count."""
    # word_scores: index 0=>word, 1=>count
    return heapq.nlargest(n, word_scores, key=lambda x: x[1])

try:
    # islice allows you to get some # of values out of a generator.
    for k, tweet in enumerate(islice(tweet_generator(), 1000)):
        for word in tweet.lower().split():  # lowercase, split by whitespace
            if word not in stop:  # avoid uninteresting words
                counter[word] += 1
        if k % DISPLAY_EVERY == (DISPLAY_EVERY - 1):  # only update text every so often
            # \r will overwrite updates, rather than listing them out
            # one on each newline
            sys.stdout.write("\r" + str(nlargest(10, counter.items())))
except KeyboardInterrupt:
    pass  # allow ctrl-c to exit the loop
finally:
    # Demo to show that Pandas has bar graphs...
    # ...and that seaborn makes it pretty!
    df = pd.DataFrame(nlargest(10, counter.items()), columns=['words', 'count'])
    df.set_index('words').plot(kind='bar')

counter = Counter()
try:
    for k, tweet in enumerate(islice(tweet_generator(), 1000)):
        for word in tweet.lower().split():
            if word not in stop and 'http' not in word:
                counter[word] += 1
        if k % DISPLAY_EVERY == (DISPLAY_EVERY - 1):
            wordcloud = WordCloud().fit_words(counter.items())
            plt.axis("off")
            display.clear_output(wait=True)
            plt.imshow(wordcloud)
            display.display(plt.gcf())
except KeyboardInterrupt:
    pass

BATCH_SIZE = 20
CLUSTER_SIZE = 4

kmeans = MiniBatchKMeans(n_clusters=CLUSTER_SIZE)

def batch(iterable, size):
    """ batch("ABCDEFG", 3) -> ABC DEF G """
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([batchiter.next()], batchiter)

with open("dictnostops.txt") as fh:
    words = [line.strip() for line in fh.readlines()]
    word_to_index = { word: k for k, word in enumerate(words) }

def wordclouds(wordcounts):
    wordclouds = [WordCloud().fit_words(counts) for counts in wordcounts]
    fig, axes = plt.subplots(2,2)
    display.clear_output(wait=True)
    for k, (ax, wordcloud) in enumerate(zip(axes.flatten(), wordclouds)):
        ax.axis("off")
        ax.imshow(wordcloud)
        ax.set_title("Topic %d" % k)
    display.display(fig)
    fig.clear()

try:
    for tweets in batch(islice(tweet_generator(), 1000), BATCH_SIZE):
        mat = sp.sparse.dok_matrix((BATCH_SIZE, len(words)))
        for row, tweet in enumerate(tweets):
            for word in tweet.lower().split():
                if word in word_to_index:
                    mat[row, word_to_index[word]] = 1.
        kmeans.partial_fit(mat.tocsr())
        wordcounts = [
            nlargest(50, zip(words, kmeans.cluster_centers_[i]))
            for i in xrange(kmeans.n_clusters)
        ]
        wordclouds(wordcounts)
except KeyboardInterrupt:
    pass

K = 4
D = 1e9
BATCH_SIZE = 20
olda = onlineldavb.OnlineLDA(words, K, D, 1./K, 1./K, 1024., 0.7)

try:
    for tweets in batch(islice(tweet_generator(), 1000), BATCH_SIZE):
        olda.update_lambda(list(tweets))
        wordclouds(olda.topic_words(50))
except KeyboardInterrupt:
    pass


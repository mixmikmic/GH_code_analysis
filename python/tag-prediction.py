get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

from collections import Counter

import numpy as np
np.random.seed(3)
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

fname = '/Users/thead/Downloads/Train.csv'
df = pd.read_csv(fname, nrows=100000, index_col='Id', engine='c')

df.head()

df.Tags = df.Tags.map(lambda x: x.split())

def encode_tags(tags, n_tags=40):
    tags_ = Counter()
    for v in tags:
        tags_.update(v)

    keys = list(sorted(v[0] for v in tags_.most_common(n_tags)))

    encoded = np.zeros((len(tags), n_tags))
    for i,row in enumerate(tags):
        for tag in row:
            if tag in keys:
                j = keys.index(tag)
                encoded[i, j] = 1
                
    return encoded
    
encode_tags(df.Tags[:10])

from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_code = False
        self.text = []
        
    def handle_starttag(self, tag, attrs):
        if tag == 'code':
            self.in_code = True

    def handle_endtag(self, tag):
        if tag == 'code':
            self.in_code = False

    def handle_data(self, data):
        if not self.in_code:
            self.text.append(data)


def clean_body(body):
    extractor = TextExtractor()
    extractor.feed(body)
    return ' '.join(extractor.text)


df['CleanBody'] = df.Body.map(clean_body)
        
tfidf = TfidfVectorizer(stop_words='english')

df['Text'] = df.apply(lambda x: x.Title +' '+x.CleanBody, axis=1)

df.head()

tags = Counter()
for v in df.Tags:
    tags.update(v)
    
tags.most_common(40)

len(tags.keys())

plt.hist(df.Tags.apply(len), range=(0,5), bins=5)

df.Tags.apply()





from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

import pyLDAvis
pyLDAvis.enable_notebook()

#dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                             remove=('headers', 'footers', 'quotes'))


cats = ['talk.religion.misc', 'alt.atheism', 'comp.graphics', 'sci.med', 'sci.space']
dataset = fetch_20newsgroups(subset='train', categories=cats,
                             shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))

Counter(dataset.target)

dataset.target_names

def norm(a):
    a = np.asarray(a)
    return a/(a.sum(axis=1)[:,np.newaxis])

def norm_(a):
    a = np.asarray(a)
    return a/a.sum()

vect = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
lda = LatentDirichletAllocation(n_topics=10)

docs = dataset['data']
docs = df.Text[:1000]

vectorised = vect.fit_transform(docs)
doc_topic_prob = lda.fit_transform(vectorised)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
print_top_words(lda, vect.get_feature_names(), 15)

# topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency,
opts = dict(vocab=vect.get_feature_names(),
            doc_topic_dists=norm(doc_topic_prob),
            doc_lengths=np.array((vectorised != 0).sum(1)).squeeze(),
            topic_term_dists=norm(lda.components_),
            term_frequency=norm_(vectorised.sum(axis=0).tolist()[0]),)

import warnings
warnings.filterwarnings('ignore')
pyLDAvis.prepare(**opts, mds='tsne')




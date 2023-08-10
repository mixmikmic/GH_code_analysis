get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

from html.parser import HTMLParser

import numpy as np
np.random.seed(3)
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

import pyLDAvis
pyLDAvis.enable_notebook()

fname = '/Users/thead/Downloads/Train.csv'
df = pd.read_csv(fname, nrows=100000, index_col='Id', engine='c')

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
df['Text'] = df.apply(lambda x: x.Title +' '+x.CleanBody, axis=1)
df.drop(['Tags', 'Body', 'Title', 'CleanBody'], axis=1, inplace=True)
df = df.iloc[np.random.permutation(len(df))]
df.head()

# normalise entries in an array so they sum to one
def norm(a):
    a = np.asarray(a)
    return a/(a.sum(axis=1)[:,np.newaxis])

def norm_(a):
    a = np.asarray(a)
    return a/a.sum()

# This is where the magic happens!
vect = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
lda = LatentDirichletAllocation(n_topics=4, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=4)

# I am impatient, so in the interest of
# CPU time consider just the first 30000 questions
docs = df.Text[:30000]


vectorised = vect.fit_transform(docs)
doc_topic_prob = lda.fit_transform(vectorised)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % (topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
print_top_words(lda, vect.get_feature_names(), 15)

topic = doc_topic_prob[:, 3] > 0.9

for n, doc in enumerate(docs[topic]):
    print(doc)
    print('~' * 80)
    if n > 10:
        break

order = np.argsort(doc_topic_prob[:, 0], axis=0)
dd = np.zeros_like(doc_topic_prob)
for i in range(6):
    dd[:, i] = doc_topic_prob[order, i]
plt.imshow(dd[:10, :], aspect='normal', interpolation='nearest', cmap='Blues')

import seaborn as sns

fg = pd.DataFrame(doc_topic_prob, columns=["one", 'two', 'three',
                                           'four', 
                                           #'five', 'six', 'seven'
                                          ])
fg['topic'] = np.argmax(doc_topic_prob, axis=1)
sns.pairplot(fg.sample(1000), hue='topic')



opts = dict(vocab=vect.get_feature_names(),
            doc_topic_dists=norm(doc_topic_prob),
            doc_lengths=np.array((vectorised != 0).sum(1)).squeeze(),
            topic_term_dists=norm(lda.components_),
            term_frequency=norm_(vectorised.sum(axis=0).tolist()[0]),)
import warnings
warnings.filterwarnings('ignore')
pyLDAvis.prepare(**opts, mds='tsne')




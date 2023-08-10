import pandas as pd

df = pd.read_csv('data/title_abstract_doi.csv')

df.head()

len(df)

import numpy as np

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import BallTree, KDTree

STEMMER = PorterStemmer()
TOKENIZER = RegexpTokenizer(r'\w+')

class ContentRx(object):
    """
    A simple class to implement a scikit-learn-like API,
    and to hold the data.
    """
    def __init__(self,
                 components=100,
                 return_scores=True,
                 metric='euclidean',
                 centroid='median',
                 ngram_range=(1,2),   # Can be very slow above (1,2)
                 ignore_fewer_than=0, # ignore words fewer than this
                 ):
        self.components = components
        self.return_scores = return_scores
        self.centroid = centroid
        self.metric = metric
        self.ngram_range = ngram_range
        self.ignore_fewer_than = ignore_fewer_than
        
    def _preprocess(self, text):
        """
        Stem and tokenize a piece of text (e.g. an abstract).
        """
        out = [STEMMER.stem(token) for token in TOKENIZER.tokenize(text)]
        return ' '.join(out)

    def fit(self, data):
        """
        Algorithm for latent semantic analysis:
        * Create a tf-idf (e.g. unigrams and bigrams) for each doc.
        * Compute similarity with sklearn pairwise metrics.
        * Get the 100 most-similar items.
        """
        data = [self._preprocess(item) for item in data]

        # Build LSA pipline: TF-IDF then normalized SVD reduction.
        tfidf = TfidfVectorizer(ngram_range=self.ngram_range,
                                min_df=self.ignore_fewer_than,
                                stop_words='english',
                                )
        svd = TruncatedSVD(n_components=self.components)
        normalize = Normalizer(copy=False)
        lsa = make_pipeline(tfidf, svd, normalize)
        self.X = lsa.fit_transform(data)

        # Build and store distance tree.
        # metrics: see BallTree.valid_metrics
        self.tree = KDTree(self.X, metric=self.metric)

        return

    def recommend(self, likes, n_recommend=10):
        """
        Makes a recommendation.
        """
        # Make the query from the input document idxs.
        # Science Concierge uses Rocchio algorithm,
        # but I don't think I care about 'dislikes'.
        vecs = np.array([self.X[idx] for idx in likes])
        q = np.mean(vecs, axis=0).reshape(1, -1)

        # Get the matches and their distances.
        dist, idx = self.tree.query(q, k=n_recommend+len(likes))
        
        # Get rid of the original likes, which may or may not be in the result.
        ind, dist = zip(*[(i, d)
                          for d, i in zip(np.squeeze(dist), np.squeeze(idx))
                          if i not in likes])
        
        # If the likes weren't in the result, we remove the most distant results.
        if self.return_scores:
            return list(ind)[:n_recommend], list(1 - np.array(dist))[:n_recommend]
        return list(ind)[:n_recommend]

crx = ContentRx(ngram_range=(1,2))

crx.fit(df.abstract)

s = [i for i, t in enumerate(df.title) if 'spectral decomp' in t.lower()]
s

df.title[79], df.title[127]

idx, scores = crx.recommend(likes=s, n_recommend=10)

idx

df.iloc[idx]

for i, s in zip(idx, scores):
    print('{:.1f}'.format(100*s).rjust(5), df.title[i])


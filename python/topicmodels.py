from bokeh.plotting import figure
from bokeh.io import output_notebook

# Load Bokeh for visualization
output_notebook()

import nltk 
import string
import pickle
import gensim

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

CORPUS_PATH = "/Users/benjamin/Repos/git/minke/fixtures/tiny_tagged"
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

class PickledCorpus(CorpusReader, CategorizedCorpusReader):
    
    def __init__(self, root, fileids=PKL_PATTERN, cat_pattern=CAT_PATTERN):
        CategorizedCorpusReader.__init__(self, {"cat_pattern": cat_pattern})
        CorpusReader.__init__(self, root, fileids)
        
        self.punct = set(string.punctuation) | {'“', '—', '’', '”', '…'}
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.wordnet = nltk.WordNetLemmatizer() 
    
    def _resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids
    
    def lemmatize(self, token, tag):
        token = token.lower()
        
        if token not in self.stopwords:
            if not all(c in self.punct for c in token):
                tag =  {
                    'N': wn.NOUN,
                    'V': wn.VERB,
                    'R': wn.ADV,
                    'J': wn.ADJ
                }.get(tag[0], wn.NOUN)
                return self.wordnet.lemmatize(token, tag)
    
    def tokenize(self, doc):
        # Expects a preprocessed document, removes stopwords and punctuation
        # makes all tokens lowercase and lemmatizes them. 
        return list(filter(None, [
            self.lemmatize(token, tag)
            for paragraph in doc 
            for sentence in paragraph 
            for token, tag in sentence 
        ]))
    
    def docs(self, fileids=None, categories=None):
        # Resolve the fileids and the categories
        fileids = self._resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield self.tokenize(pickle.load(f))

# Create the Corpus Reader
corpus = PickledCorpus(CORPUS_PATH)

# Create the lexicon from the corpus 
lexicon = gensim.corpora.Dictionary(corpus.docs())
docvecs = [lexicon.doc2bow(doc) for doc in corpus.docs()]

# Create the LDA model from the docvecs corpus 
model = gensim.models.LdaModel(docvecs, id2word=lexicon, alpha='auto', num_topics=10)

tfidf = gensim.models.TfidfModel(docvecs, id2word=lexicon, normalize=True)

tfidfvecs = [
    tfidf[doc] for doc in docvecs
]

# Write the model and the lexicon to disk 
lexicon.save('data/lexicon.dat')
model.save('data/lda_model.dat')
model.save('data/tfidf_model.dat')

tokens = [
    " ".join([
        lexicon.id2token[tid]
        for tid, freq in sorted(doc, key=itemgetter(1))[:10]
    ])
    for doc in tfidfvecs
]

import numpy as np
from gensim.matutils import sparse2full

# Get document bag of words vectors as a full numpy array. 
docarr = np.array([sparse2full(vec, len(lexicon)) for vec in tfidfvecs])

from operator import itemgetter 

topics = [
    max(model[doc], key=itemgetter(1))[0]
    for doc in docvecs
]

# Generate PCA 
from sklearn.decomposition import PCA 

pca = PCA(n_components=2)
pcavecs = pca.fit_transform(docarr)

from bokeh.palettes import brewer
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool

cmap = {
    i: brewer['Paired'][10][i]
    for i in range(10)
}

source = ColumnDataSource(
        data=dict(
            x=pcavecs[:, 0],
            y=pcavecs[:, 1],
            w=tokens,
            t=topics,
            c=[cmap[t] for t in topics],
        )
    )

hover = HoverTool(
        tooltips=[
            ("Words", "@w"),
            ("Topic", "@t"),
        ]
    )

plt = figure(title="PCA Decomposition of BoW Space", width=960, height=540, tools="pan,box_zoom,reset,resize,save")
plt.add_tools(hover)
plt.scatter('x', 'y', source=source, marker='circle_x', line_color='c', fill_color='c', fill_alpha=0.5, size=9)

show(plt)

from collections import defaultdict 

tsize = defaultdict(int)
for doc in docvecs:
    for tid, prob in model[doc]:
        tsize[tid] += prob

tvecs = np.array([
    sparse2full(model.get_topic_terms(tid, len(lexicon)), len(lexicon)) for tid in range(10)
])

tpca  = PCA(n_components=2)
pcatvecs = tpca.fit_transform(tvecs)

tsource = ColumnDataSource(
        data=dict(
            x=pcatvecs[:, 0],
            y=pcatvecs[:, 1],
            w=[model.print_topic(tid, 10) for tid in range(10)],
            c=brewer['Spectral'][10],
            r=[tsize[idx]/300000.0 for idx in range(10)],
        )
    )

hover = HoverTool(
        tooltips=[
            ("Words", "@w"),
        ]
    )

plt = figure(title="Topic Model Decomposition", width=960, height=540, tools="pan,box_zoom,reset,resize,save")
plt.add_tools(hover)
plt.scatter('x', 'y', radius='r', source=tsource, marker='circle', line_color='c', fill_color='c', fill_alpha=0.85, size=9)

show(plt)


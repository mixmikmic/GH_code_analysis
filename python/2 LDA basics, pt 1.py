# import and setup modules we'll be using in this notebook
import logging
import itertools
import os
import pickle

import numpy as np
import gensim

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
# from gensim.parsing.preprocessing import STOPWORDS
from cltk.stop.greek.stops import STOPS_LIST

STOPS_LIST = [simple_preprocess(stop, deacc=True)[0] for stop in STOPS_LIST if len(simple_preprocess(stop, deacc=True)) > 0]

def tokenize(text):
    # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
    tokens = [token for token in simple_preprocess(text, deacc=True)]
    return [token for token in tokens if token not in STOPS_LIST]
    

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens

def iter_tlg(tlg_dir):
    file_names = os.listdir(tlg_dir)
    for file_name in file_names:
        file_path = os.path.join(tlg_dir, file_name)
        with open(file_path) as file_open:
            file_read = file_open.read()
        tokens = tokenize(file_read)
        # ignore short docs
        if len(tokens) < 50:
            continue
        yield file_name, tokens

#stream = iter_wiki('./data/simplewiki-20140623-pages-articles.xml.bz2')

tlg_preprocessed = os.path.expanduser('~/cltk_data/greek/text/tlg/plaintext/')
stream = iter_tlg(tlg_preprocessed)

for title, tokens in itertools.islice(iter_tlg(tlg_preprocessed), 8):
    print(title, tokens[:10])  # print the article title and its first ten tokens

doc_stream = (tokens for _, tokens in iter_tlg(tlg_preprocessed))

get_ipython().magic('time id2word_tlg = gensim.corpora.Dictionary(doc_stream)')
print(id2word_tlg)

# this cutoff might lose too much info, we'll see
# ignore words that appear in less than 20 documents or more than 10% documents
id2word_tlg.filter_extremes(no_below=20, no_above=0.1)
print(id2word_tlg)

doc = "περὶ ποιητικῆς αὐτῆς τε καὶ τῶν εἰδῶν αὐτῆς, ἥν τινα δύναμιν ἕκαστον ἔχει, καὶ πῶς δεῖ συνίστασθαι τοὺς μύθους [10] εἰ μέλλει καλῶς ἕξειν ἡ ποίησις, ἔτι δὲ ἐκ πόσων καὶ ποίων ἐστὶ μορίων, ὁμοίως δὲ καὶ περὶ τῶν ἄλλων ὅσα τῆς αὐτῆς ἐστι μεθόδου, λέγωμεν ἀρξάμενοι κατὰ φύσιν πρῶτον ἀπὸ τῶν πρώτων."
doc = ' '.join(simple_preprocess(doc))
bow = id2word_tlg.doc2bow(tokenize(doc))
print(bow)

print(id2word_tlg[6880], id2word_tlg[12323])

# Save for reuse
# can also use `id2word_tlg.save('~/cltk_data/user_data/tlg_bow_id2word.dict')`
with open(os.path.expanduser('~/cltk_data/user_data/tlg_bow_id2word.dict'), 'wb') as file_open:
    pickle.dump(id2word_tlg, file_open)

class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs

class TLGCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """Yield each document in turn, as a list of tokens (unicode strings).
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_tlg(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs

# create a stream of bag-of-words vectors
tlg_corpus = TLGCorpus(tlg_preprocessed, id2word_tlg)
vector = next(iter(tlg_corpus))
print(vector)  # print the first vector in the stream

# get titles, save to disk
tlg_corpus = TLGCorpus(tlg_preprocessed, id2word_tlg)
for item in tlg_corpus:
    print(type(item))
    input()



# what is the most common word in that first article?
most_index, most_count = max(vector, key=lambda _tuple: _tuple[1])
print(id2word_tlg[most_index], most_count)

from gensim.corpora.mmcorpus import MmCorpus

# Save BoW
user_dir = os.path.expanduser('~/cltk_data/user_data/')
try:
    os.makedirs(user_dir)
except FileExistsError:
    pass
bow_path = os.path.join(user_dir, 'bow_lda_gensim.mm')

get_ipython().magic('time gensim.corpora.MmCorpus.serialize(bow_path, tlg_corpus)')

mm_corpus = gensim.corpora.MmCorpus(bow_path)
print(mm_corpus)

print(next(iter(mm_corpus)))


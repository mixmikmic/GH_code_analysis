import nbimporter
from corpora import FileStream
from indexing import Tokenizer, DBIndex
import numpy as np
from IPython.display import clear_output

folder = 'data/wikisearch/brat_20'
corpus = FileStream(folder, file_ext='txt')
tokenizer = Tokenizer(preserve_case=False)
I = DBIndex('inforet')
c = 'wikisearch'

from gensim import corpora
from gensim import matutils

dictionary = corpora.Dictionary(tokens for tokens in I.stream(c))

print dictionary

print dictionary.token2id.items()[:6]

print dictionary[22]

print dictionary.doc2bow(['man', 'work', 'business', 'to', 'business'])

class Wiki(object):
    
    def __init__(self, collection, _index, dictionary):
        self.collection = collection
        self.i = _index
        self.dict = dictionary
        
    def __iter__(self):
        for tokens in self.i.stream(self.collection):
            yield self.dict.doc2bow(tokens)

corpus = Wiki(c, I, dictionary)

corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)

mm = corpora.MmCorpus('/tmp/corpus.mm')

print mm

matrix = matutils.corpus2dense(mm, num_terms=9327)

print matrix




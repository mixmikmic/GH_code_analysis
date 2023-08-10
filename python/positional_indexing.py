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

for i, doc_id in enumerate(corpus.docs):
    doc = corpus.doc(doc_id)
    words, lemmata = tokenizer.pattern_processing(doc, lemmata=True)
    words = tokenizer.remove_punctuation(words)
    tokens = tokenizer.remove_punctuation(lemmata)
    I.index(c, doc_id, words, tokens)
    print i+1, 'of', len(corpus.docs), 'indexed'
    clear_output(wait=True)

idf = I.idf_to_vec(c)

tf = I.tf_to_matrix(c)

tfidf = tf * idf

print tfidf

docs, tokens = I.docs(c), I.tokens(c)

print tfidf[docs.index('37968451.txt'), tokens.index('be')]
print tfidf[docs.index('37968451.txt'), tokens.index('country')]
print tfidf[0, tokens.index('be')]
print tfidf[0, tokens.index('country')]

print corpus.doc('37968451.txt')

print corpus.doc(docs[0])




# Preprocessing
# Gensim requires list of lists of Unicode 8 strings as an input. Since we have a small collection, we are fine with loading everything into memory.
import re
doc_list= []
with open('../nfcorpus/raw/doc_dump.txt', 'r') as rf1:
    for line in rf1:
        l = re.sub("MED-.*\t", "",line).lower().strip('\n').split()
        doc_list.append(l) 
len(doc_list) # TODO: Report this in project report

from gensim import models
# step 1: train the detector with
phrases = models.phrases.Phrases(doc_list, min_count=2) # phrases have to occur at least two times
# step 2: create a Phraser object to transform any sentence 
bigram = models.phrases.Phraser(phrases)

#little sanity check to see if it has worked: breast cancer should be detected as a collocation
bigram['Exhibits', 'a', 'high', 'risk ' ,'of' , 'breast', 'cancer']

import gensim
#word2vec = models.Word2Vec(bigram[doc_list],min_count=1, workers=4)
word2vec = models.Word2Vec(doc_list,min_count=1, workers=4)
# also tried with skipgram, produces same vocablary, also set min_count to zero, produces same vocabulary
word2vec.save('our_word2vec')

''' 
# free RAM as recommended in the docs and if we top training
word2vec_vectors = word2vec.wv
del (word2vec.wv)
'''

[i in word2vec.wv for i in [ 'of', 'by', 'the', 'and','.',',','%','$','2', '23', '234','X','Can']]

[i in word2vec.wv for i in ['describe', 'described', 'describes', 'describing']]

#but we will live with this as they are subjectively very similar and this goes beyond the scope of our topic
word2vec.wv.similarity('describe', 'described')

# as expected we created a 64585 dimensional vocabulary, each word being described by a 100 dimensional dense vector
word2vec.wv.vectors.shape # 89269 if allwing for bigrams

import pandas as pd
inverted_index = pd.read_pickle('../0_Collection_and_Inverted_Index/pickle/inverted_index.pkl')

# BoW vocabulary 
len(inverted_index.index)

#overlap between two sets
overlap=set()
no_overlap=set()
for word in list(inverted_index.index):
    if word in word2vec.wv.vocab: # ... in  word2vec.wv  : returns the same
        overlap.add(word)
    else: 
        no_overlap.add(word)
len(no_overlap)

#this is only hard to explain why these words are in the smaller BoW representation and not the vocabulary obtained from WEs
no_overlap

gensim.models.fasttext.FAST_VERSION > -1 # make sure that you are using Cython backend

import gensim
#fasttext= gensim.models.FastText(bigram[doc_list], min_count= 1, min_n= 3, max_n=12)
fasttext= gensim.models.FastText(doc_list, min_count= 1, min_n= 3, max_n=12)

fasttext.save('our_fasttext')

''' 
# free RAM as recommended in the docs
fasttext_vectors = fasttext.wv
del (fasttext.wv)
type(fasttext_vectors)
'''

# you don't want to recompute the FastText vectors since it takes quite long
# this loads the whole model, (not only the vectors)
fasttext= gensim.models.FastText.load('our_fasttext')

# put this in presentation: this is why W2V with Subwords are cool..
fasttext.wv.similarity('breawe caner', 'breast cancer') # this is the primary use case: out of vocab predictions

# overlap between two sets
# fasttext produces the result we expect, word2vec however not
overlap=set()
no_overlap=set()
for word in list(inverted_index.index):
    if word in fasttext.wv:
        overlap.add(word)
    else: 
        no_overlap.add(word)
len(no_overlap)

len(overlap)

no_overlap

fasttextword2vec= gensim.models.FastText(doc_list, min_count= 1, word_ngrams=0)

fasttext.save('our_fasttextword2vec')

# overlap between two sets
# fasttext produces the result we expect, word2vec however not
overlap=set()
no_overlap=set()
for word in list(inverted_index.index):
    if word in fasttextword2vec.wv:
        overlap.add(word)
    else: 
        no_overlap.add(word)
len(no_overlap)

no_overlap

fasttextword2vec.vocabulary.max_vocab_size

len(fasttextword2vec.wv.vocab)

len(fasttextword2vec.wv.vocab)

len(word2vec.wv.vocab)

# fasttext and fasttextword2vec have same vocabulary
overlap=set()
no_overlap=set()
for word in fasttextword2vec.wv.vocab:
    if word in fasttext.wv.vocab:
        overlap.add(word)
    else: 
        no_overlap.add(word)
len(no_overlap)

#obviously the vocabulary is the same in all three cases...
no_overlap=set()
for word in fasttextword2vec.wv.vocab:
    if word in word2vec.wv.vocab:
        overlap.add(word)
    else: 
        no_overlap.add(word)
len(no_overlap)

fasttext.wv.num_ngram_vectors

fasttextword2vec.wv.num_ngram_vectors # > 64585: makes sense, that there are more n-grams words in the vocabulary 



# since we are not using any subword information (fasttext ngrams for out of vocabulary words), we can import the embeddings as easy as follows
from gensim.models.keyedvectors import KeyedVectors
word2vec_wiki = KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec")
fasttext_wiki = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subword.vec")
fasttext_commoncrawl = KeyedVectors.load_word2vec_format("crawl-300d-2M.vec")

# whave a 300 dimensional dense vector for all models
300==len(word2vec_wiki.get_vector('cancer'))==len(fasttext_wiki.get_vector('cancer'))==len(fasttext_commoncrawl.get_vector('cancer'))

import pandas as pd
inverted_index=pd.read_pickle('inverted_index.pkl')
#overlap with vocabulary from Word2Vec emdeddings generated from English Wikipeda
overlap=set()
no_overlap=set()
for word in list(inverted_index.index):
    if word in word2vec_wiki.wv:
        overlap.add(word)
    else: 
        no_overlap.add(word)
len(no_overlap)

#overlap with vocabulary from fasttext emdeddings generated from English Wikipeda
overlap=set()
no_overlap=set()
for word in list(inverted_index.index):
    if word in fasttext_wiki.wv:
        overlap.add(word)
    else: 
        no_overlap.add(word)
len(no_overlap)

#overlap with vocabulary from fasttext emdeddings generated from commoncrawl
overlap=set()
no_overlap=set()
for word in list(inverted_index.index):
    if word in fasttext_wiki.wv:
        overlap.add(word)
    else: 
        no_overlap.add(word)
len(no_overlap)

len(overlap)

# we would throw away good domain-specific candidates when only looking at the union of both vocabularies
no_overlap

len(fasttext.wv.vectors)==len(word2vec.wv.vectors)


import pandas as pd
import numpy as np
import pickle
#We need this line to find the collection_vocabulary.py here, else we cannot load the col.pkl object
import sys
sys.path.append('../0_Collection_and_Inverted_Index/')
with open('../0_Collection_and_Inverted_Index/pickle/col.pkl', 'rb') as input:
    col = pickle.load(input)
inverted_index = pd.read_pickle('../0_Collection_and_Inverted_Index/pickle/inverted_index.pkl')

tfidf=pd.read_pickle('pickle/tfidf.pkl')

# Preprocessing
# Gensim requires list of lists of Unicode 8 strings as an input. Since we have a small collection, 
# we are fine with loading everything into memory.
import re
doc_list= []
with open('../nfcorpus/raw/doc_dump.txt', 'r', encoding='utf-8') as rf1:
    for line in rf1:
        l = re.sub("MED-.*\t", "",line).lower().strip('\n').split()
        doc_list.append(l) 
len(doc_list)

import gensim
gensim.models.fasttext.FAST_VERSION > -1 # make sure that you are using Cython backend

#Run this to create a fasttext model of our documents
#Name fasttest basically means Word2Vec with subword information
fasttext= gensim.models.FastText(doc_list, min_count= 1, min_n= 3, max_n=12)
fasttext.save('pickle/our_fasttext')

#Same as above, run this to compute the model, or run next cell to load it (if it exists on disk already)
word2vec= gensim.models.FastText(doc_list, min_count= 1, word_ngrams=0)
word2vec.save('pickle/our_fasttextword2vec')

# To save time, load the models, if they already exist.
# This loads the whole models (not only the vectors).
fasttext = gensim.models.FastText.load('pickle/our_fasttext') 
word2vec = gensim.models.FastText.load('pickle/our_fasttextword2vec')

# Word2Vec Embeddings with Subword Information, 100-d dense vector 
fasttext_embeddings_list=[]
words_not_covered_in_fasttext=[]
for word in inverted_index.index:
    try:
        fasttext_embeddings_list.append(fasttext.wv.get_vector(word))
    except:
        words_not_covered_in_fasttext.append(word)
        fasttext_embeddings_list.append(np.zeros(100)) # for those 3 OOV we insert an array consisting of zeros
fasttext_embeddings=pd.Series(fasttext_embeddings_list,index=inverted_index.index)
fasttext_embeddings.to_pickle('pickle/fasttext_embeddings.pkl')
fasttext_embeddings.head()

#Word2Vec Embeddings, 100-d dense vector
word2vec_embeddings_list=[]
words_not_covered_in_word2vec=[]
for word in inverted_index.index:
    try:
        word2vec_embeddings_list.append(word2vec.wv.get_vector(word))
    except:
        words_not_covered_in_word2vec.append(word)
        word2vec_embeddings_list.append(np.zeros(100)) # for those 3 OOV we insert an array consisting of zeros
word2vec_embeddings=pd.Series(word2vec_embeddings_list,index=inverted_index.index)
word2vec_embeddings.to_pickle('pickle/word2vec_embeddings.pkl')
word2vec_embeddings.head()

fasttext_embeddings = pd.read_pickle('pickle/fasttext_embeddings.pkl')
word2vec_embeddings = pd.read_pickle('pickle/word2vec_embeddings.pkl')

def get_weighted_embeddings(embeddings, tfidf_embed):
    sum_of_tfidf_weights=tfidf_embed.sum(axis=0)#vector containing the normalizing constant for each doc
    embeddings_dict={}
    # we have to make use of the following workaround to avoid memory errors
    # 1. calculate 100d embeddings vector for each doc/query and store it in dictionary
    # 2. recreate a a dataframe containg the embeddings for all docs/queries from the dictionary
    for doc in tfidf_embed.columns:
        if doc not in embeddings_dict.keys():
            embedding=(tfidf_embed[doc].mask(tfidf_embed[doc]!=0, other=(tfidf_embed[doc]*embeddings)).sum(axis=0))/sum_of_tfidf_weights[doc]
            embeddings_dict[doc]=embedding
        else:
            print('calculated embeddings successfully and stored them in dictionary')
    weighted_embedding = pd.DataFrame.from_dict(embeddings_dict)
    return weighted_embedding

documents_fasttext = get_weighted_embeddings(fasttext_embeddings, tfidf)

#Let's save those again, as computing them might take a while
documents_fasttext.to_pickle('pickle/documents_fasttext.pkl')

documents_word2vec= get_weighted_embeddings(word2vec_embeddings, tfidf)

#Save them as well
documents_word2vec.to_pickle('pickle/documents_word2vec.pkl')

#put this in report 
words_not_covered_in_word2vec

#put this in report 
words_not_covered_in_fasttexttext

#put this in report 
len(word2vec.wv.vectors)

#put this in report 
len(fasttext.wv.vectors)


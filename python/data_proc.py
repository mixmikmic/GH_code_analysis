import os
import numpy as np
import pandas as pd
import ast
import re
import csv

import nltk
#from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
ps = PorterStemmer()
lemma = WordNetLemmatizer()

from sklearn.feature_extraction.text import CountVectorizer


import gensim
from gensim import corpora, models
os.chdir('~/Codes/DL - Topic Modelling')

dir_src = os.path.join(os.getcwd(), 'data/raw_20news/20news-18828')

dir_src_classes = list( map(lambda x: os.path.join(dir_src, x ), os.listdir(dir_src)) )

dat = []
dat_y = []
dat_y_cat = []

for i in range(0,len(dir_src_classes)):
    
    print('Currently loading the following topic (iteration ' + str(i) + '):\n \t' + dir_src_classes[i])
    dir_src_classes_file = list( map(lambda x: os.path.join(dir_src_classes[i], x), os.listdir(dir_src_classes[i])) )
    
    for ii in range(0, len(dir_src_classes_file)):
        
        dat_y.append(i)
        
        with open(dir_src_classes_file[ii], encoding='ISO-8859-1') as file:
            dat.append(file.read().replace('\n', ' '))

# mapping sub topics 0-19 to broader topics 0-5
y_map = [3,0,0,0,0,0,5,1,1,1,1,2,2,2,2,3,4,4,4,3]
dat_y2 = [y_map[idx] for idx in dat_y]

#export data
pd.DataFrame( { '_label_granular' : dat_y, 
                '_label_overview' : dat_y2,
                'document' : [' '.join(re.sub('[^a-zA-Z]+', ' ', doc).strip().split()) for doc in dat]}). \
                to_csv('data/raw_20news/20news.csv',
                    index=False, sep=',', encoding='ISO-8859-1')

print('------- Data cleaning -------')                
stopwords_en = stopwords.words('english')
dat_clean = []
for i in range(len(dat)):

    ''' tokenization and punctuation removal '''
    # uses nltk tokenization - e.g. shouldn't = [should, n't] instead of [shouldn, 't]
    tmp_doc = nltk.tokenize.word_tokenize(dat[i].lower())
    
    # split words sperated by fullstops
    tmp_doc_split = [w.split('.') for w in tmp_doc if len(w.split('.')) > 1]
    # flatten list
    tmp_doc_split = [i_sublist for i_list in tmp_doc_split for i_sublist in i_list]
    # clean split words
    tmp_doc_split = [w for w in tmp_doc_split if re.search('^[a-z]+$',w)]
    
    # drop punctuations
    tmp_doc_clean = [w for w in tmp_doc if re.search('^[a-z]+$',w)]
    tmp_doc_clean.extend(tmp_doc_split)

    ''' stop word removal'''
    tmp_doc_clean_stop = [w for w in tmp_doc_clean if w not in stopwords_en]
    #retain only words with 2 characters or more
    tmp_doc_clean_stop = [w for w in  tmp_doc_clean_stop if len(w) >2]
    
    ''' stemming (using the Porter's algorithm)'''
    tmp_doc_clean_stop_stemmed = [ps.stem(w) for w in tmp_doc_clean_stop]
    dat_clean.append(tmp_doc_clean_stop_stemmed)
    
    #print progress
    if i % 100 == 0: print( 'Current progress: ' + str(i) + '/' + str(len(dat)) )

dat = pd.read_csv('data/clean_20news.csv', sep=",")


docs = [ast.literal_eval(doc) for doc in dat['document'].tolist()]

all_words = [word for doc in docs for word in doc]
pd_all_words = pd.DataFrame({'words' : all_words})
pd_unq_word_counts = pd.DataFrame({'count' : pd_all_words.groupby('words').size()}).reset_index().sort('count', ascending = False)

# follow's research paper's top 2000 vocabulary (previously only took data with counts of words more than 150)
pd_unq_word_counts_filtered = pd_unq_word_counts.head(2000)
list_unq_word_filtered = list( pd_unq_word_counts_filtered.ix[:,0] )
len(list_unq_word_filtered)

vec = CountVectorizer(input = 'content', lowercase = False, vocabulary = list_unq_word_filtered)

iters = list(range(0,len(docs),500))
iters.append(len(docs))
dtm = np.array([] ).reshape(0,len(list_unq_word_filtered))
for i in range(len(iters)-1):
    dtm = np.concatenate( (dtm, list(map(lambda x: vec.fit_transform(x).toarray().sum(axis=0), docs[iters[i]:iters[i+1]] )) ), axis = 0)
    print( 'Percentage completion: ' + str( (i+1) / (len(iters)-1) ) )

colnames = list_unq_word_filtered
colnames.insert(0,'_label_')

pd.DataFrame(data = np.c_[dat['label'].values, dtm], 
             columns = colnames). \
             to_csv( 'data/dtm_2000_20news.csv', index = False)

pd.DataFrame(data = np.c_[dat_y2, dtm], 
             columns = colnames). \
             to_csv( 'data/dtm_2000_20news_6class.csv', index = False)

list_unq_word_filtered

df = pd.read_csv('data/raw_20news/20news.csv', sep=",")
df


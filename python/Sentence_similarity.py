from pymongo import MongoClient
import pandas as pd
from sklearn.externals import joblib
from multiprocessing import Pool
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRow, IndexedRowMatrix
from pyspark import SparkContext
import pandas as pd
import re
import string
from sklearn.externals import joblib

from pyspark import SparkConf, SparkContext
conf = (SparkConf()
     .set("spark.driver.extraJavaOptions", "-Xss4M")
     .set('spark.executor.memory', '4G')
     .set('spark.driver.memory', '55G')
     .set('spark.driver.maxResultSize', '10G'))
sc = SparkContext.getOrCreate(conf = conf)

#vectors = joblib.load('sentence_vectors')
id_s = joblib.load('sentence_ids')

df = pd.DataFrame({str(k):v for k, v in zip(id_s, vectors)})

#df.shape

rows = sc.parallelize(df.values)
matrix = RowMatrix(rows)

# print(matrix.numCols())
# print(matrix.numRows())

exact = matrix.columnSimilarities()

exact.numCols()

client = MongoClient()
db = client.lingbuzz

keywords = db.get_collection('keywords')

for doc in keywords.find()[:2]:
    print(doc)

sentences = db.get_collection('sentences')

sentences.find().count()

test_cases = []
for doc in keywords.find()[:2]:
    test_cases.append([id_ for id_ in doc['sentenceIDs']])

test_cases = []
for doc in keywords.find({'word':'focus'}):
    test_cases.append([id_ for id_ in doc['sentenceIDs']])

df = pd.DataFrame({k:v for k, v in zip(id_s, vectors)})

df.head()

test_cases[0]

df_test1 = df[test_cases[0]]

def create_df_cs(vectors, ids):
    """calculates relative cosine distance between two sentences and returns df with sentenceids and their distance"""
    cos_sim = cosine_similarity(np.asarray(vectors))
    df = pd.DataFrame(cos_sim, index = ids, columns = ids)
    return df

df_test1.shape

cs_test1 = create_df_cs(df_test1.values.T, list(df_test1))

cs_test1.shape

sorted(list(cs_test1.iloc[1]), reverse=True)

similar_sent = cs_test1[cs_test1.iloc[:,0] > 0.93]

similar_sent.head()

similar_sent_indexes = list(similar_sent.index)
similarity_score = similar_sent.iloc[:,0]
id_to_score = {k:v for k, v in zip(similar_sent_indexes, similarity_score)}

for sent in sentences.find({'_id': { '$in':  similar_sent_indexes }}):
    print(sent['_id'],id_to_score[sent['_id']], '\n', sent['sentence'], '\n\n')

def is_english_sentence(sent):
    """determines whether a word is English/author"""
    sentence = []
    for w in str(sent).split():
        w = str(w).lower()
        if w in authors:
            sentence.append(w)
        else: 
            try:
                w.encode(encoding='utf-8').decode('ascii')
                    # if re.sub('-', '', word).isalpha():
                        # english_words.append(re.sub('[%s]' % re.escape(string.punctuation), '', word))
                word = re.sub('[%s]' % re.escape(string.punctuation), '', w)
                if word.isalpha():
                    sentence.append(word)
            except UnicodeDecodeError:
                pass
    return sentence

def calculate_sentence_vector(words, word_to_vec, num_features = 300):
    featureVec = np.zeros((num_features,), dtype="float32")
    N = 631676
    for word in words:
        if word in word_to_vec:
            idf = np.log((1+N)/(1+len(keywords.find_one({'word':word})['sentenceIDs'])))
            featureVec = np.add(featureVec, word_to_vec[word]['vector']*idf)
    return featureVec


# voc_vectors = {}
# with open('voc_vectors.txt', 'rb') as f:
#     content = f.readlines()
# 
# for line in content:
#     line = line.decode("utf-8").split(" ", 1)
#     voc_vectors[line[0]] = {'vector': np.fromstring(line[1].strip(), sep=' ')}
#     
# authors = joblib.load('authors')
# bigrams = joblib.load('bigrams_model')
# 
# # udpate word vectors dict with sentence IDs and calculate sentence vectors
# vectors = []
# for sent in sentences.find():
#     sentence = bigrams[is_english_sentence(sent['sentence'].split())]
#     vectors.append(calculate_sentence_vector(sentence, voc_vectors))
# joblib.dump(vectors, 'sentence_vectors_idf')

df = pd.DataFrame({k:v for k, v in zip(id_s, vectors)})

def inspect_sentence_similarity(word):
    test_cases = []
    for doc in keywords.find({'word':word}):
        test_cases.append([id_ for id_ in doc['sentenceIDs']])
    df_test1 = df[test_cases[0]]    
    cs_test1 = create_df_cs(df_test1.values.T, list(df_test1))
    similar_sent = cs_test1[cs_test1.iloc[:,0] > 0.93]
    similar_sent_indexes = list(similar_sent.index)
    similarity_score = similar_sent.iloc[:,0]
    id_to_score = {k:v for k, v in zip(similar_sent_indexes, similarity_score)}
    for sent in sentences.find({'_id': { '$in':  similar_sent_indexes }}):
        print(sent['_id'],id_to_score[sent['_id']], '\n', sent['sentence'], '\n\n')

word2vec = joblib.load('word2vec')

bigrams = joblib.load('bigrams_model')

authors = joblib.load('authors')

def avg_feature_vector(words, word_to_vec, id_, num_features = 300):
    """words is list of words, num_features in dimension of vector, word_to_vec is dict with word:vector
    appends sentence ids to word_to_vec so we can quickly recover which words are in which sentences
    returns average feature vector for the sentence"""
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    for word in words:
        if word in word_to_vec:
            nwords = nwords+1
            featureVec = np.add(featureVec, word_to_vec[word]['vector'])
            word_to_vec[word]['sentenceIDs'].append(id_)
    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec

vectors = []
for sent in sentences.find():
    sentence = bigrams[is_english_sentence(sent['sentence'].split())]
    vectors.append(calculate_sentence_vector(sentence, word2vec))

with open('sentences_for_ft.txt', 'wb') as f:
    for sent in sentences.find()[:2000]:
        print(sent['sentence'])
        f.write(' '.join(is_english_sentence(sent['sentence'])).encode("utf-8"))
        f.write('\n'.encode("utf-8"))

with open('sentence_vecs_ft.txt', 'rb') as f:
    vecs_ft = f.readlines()

vecs_ft[0]

featureVec = []
for vec in vecs_ft:
    vec = re.sub('(?<!\d)[^\s0-9.-]', '', str(vec)).strip().split()
    vec_final = []
    for v in vec:
        try:
            vec_final.append(float(v))
        except:
            pass
    featureVec.append(np.array(vec_final))
featureVec = np.array(featureVec)

featureVec.shape

def create_df_rel_cs(vectors, ids):
    """calculates relative cosine distance between two sentences and returns df with sentenceids and their distance"""
    cos_sim = cosine_similarity(np.asarray(vectors))
    # sum_cs = np.sum(cos_sim, 1)[0]
    # rel_cs = cos_sim / sum_cs
    df = pd.DataFrame(cos_sim, index = ids, columns = ids)
    return df

df = create_df_rel_cs(featureVec, id_s[:2000])

df.head()

similar_sent = df[(df.iloc[:,4] < 1.1) & (df.iloc[:,4] > 0.95)]
similar_sent_indexes = list(similar_sent.index)
similarity_score = similar_sent.iloc[:,4]
id_to_score = {k:v for k, v in zip(similar_sent_indexes, similarity_score)}
for sent in sentences.find({'_id': { '$in':  similar_sent_indexes }}):
    print(sent['_id'],id_to_score[sent['_id']], '\n', sent['sentence'], '\n\n')

import spacy
nlp = spacy.load('en_core_web_sm')

def is_meaningfull(sent):
    """retains nouns, verbs, adjectives and authors only"""
    sentence = []
    for w in nlp(sent):
        if str(w.lower_) in authors:
            sentence.append(str(w.lower_))
        if w.pos_ in ['VERB' , 'ADJ' , 'NOUN']:
            try:
                str(w).encode(encoding='utf-8').decode('ascii')
                    # if re.sub('-', '', word).isalpha():
                        # english_words.append(re.sub('[%s]' % re.escape(string.punctuation), '', word))
                word = re.sub('[%s]' % re.escape(string.punctuation), '', str(w.lower_))
                if word.isalpha():
                    sentence.append(word)
            except UnicodeDecodeError:
                pass
    return sentence

with open('sentences_for_ft.txt', 'wb') as f:
    for sent in sentences.find():
        f.write(' '.join(is_meaningfull(sent['sentence'])).encode("utf-8"))
        f.write('\n'.encode("utf-8"))

with open('sentence_vecs_ft.txt', 'rb') as f:
    vecs_ft = f.readlines()

featureVec = []
for vec in vecs_ft:
    vec = re.sub('(?<!\d)[^\s0-9.-]', '', str(vec)).strip().split()
    vec_final = []
    for v in vec:
        try:
            vec_final.append(float(v))
        except:
            pass
    featureVec.append(np.array(vec_final))
featureVec = np.array(featureVec)

df = create_df_rel_cs(featureVec, id_s[:2000])

df.head(6)

similar_sent = df[(df.iloc[:,4] < 1.1) & (df.iloc[:,4] > 0.80)]
similar_sent_indexes = list(similar_sent.index)
similarity_score = similar_sent.iloc[:,4]
id_to_score = {k:v for k, v in zip(similar_sent_indexes, similarity_score)}
for sent in sentences.find({'_id': { '$in':  similar_sent_indexes }}):
    print(sent['_id'],id_to_score[sent['_id']], '\n', sent['sentence'], '\n\n')

joblib.dump(featureVec, 'featurevec')


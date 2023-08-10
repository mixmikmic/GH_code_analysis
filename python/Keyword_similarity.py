from gensim.models import Word2Vec
from gensim.models import phrases
from pymongo import MongoClient
import pandas as pd
import en_core_web_sm
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.externals import joblib
import spacy
import re
import string
from multiprocessing import Pool
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

standard_stopwords = set(list(ENGLISH_STOP_WORDS)+list(stopwords.words('english')))

client = MongoClient()
db = client.lingbuzz
keywords = db.get_collection('keywords')

voc_vectors = {}
with open('voc_vectors.txt', 'rb') as f:
    content = f.readlines()

for line in content:
    line = line.decode("utf-8").split(" ", 1)
    voc_vectors[line[0]] = {'vector': np.fromstring(line[1].strip(), sep=' '), 'sentenceIDs' : []}

# the formula used here is wrong. According to the paper it should be the sum of the similarities 
# between the top n most similar words...

# def create_df_rel_cs(vectors, ids):
#     """calculates relative cosine distance between two sentences and returns df with sentenceids and their distance"""
#     cos_sim = cosine_similarity(np.asarray(vectors))
#     sum_cs = np.sum(cos_sim, 1)[0]
#     rel_cs = cos_sim / sum_cs
#     df = pd.DataFrame(rel_cs, index = ids, columns = ids)
#     return df

def create_df_cs(vectors, ids):
    """calculates relative cosine distance between two sentences and returns df with sentenceids and their distance"""
    cos_sim = cosine_similarity(np.asarray(vectors))
    df = pd.DataFrame(cos_sim, index = ids, columns = ids)
    return df

kwords = []
keyword_vec = []
for entry in voc_vectors:
    if entry not in standard_stopwords:
        kwords.append(entry)
        keyword_vec.append(voc_vectors[entry]['vector'])

keyword_similarities = create_df_cs(keyword_vec, kwords)

keyword_similarities.head()

synonyms = []
headers = list(keyword_similarities.columns)
for i, row in keyword_similarities.head(20).iterrows():
    most_similar = sorted(row.drop(i))[-10:]
    columns = list(row.isin(most_similar))
    synonyms.append((i, list(compress(headers, columns)), most_similar))

from itertools import compress
synonyms = {}
headers = list(keyword_similarities.columns)
for i, row in keyword_similarities.iterrows():
    most_similar = [x for x in row.drop(i) if x >= 0.635]
    columns = list(row.isin(most_similar))
    synonyms[i] =  list(compress(headers, columns))

synonyms['czech']

joblib.dump(synonyms, 'similar_keywords')

similar_keywords = joblib.load('similar_keywords')

voc_vectors_dict = joblib.load('voc_vectors_dict')

id_keywords = {}
for keyword in keywords.find():
    id_keywords[keyword['word']]=keyword['_id']

for k, v in similar_keywords.items():
    keyword_ids = [id_keywords[kw] for kw in v]
    keywords.update_one({'word': k}, {'$set': {'similar_words': keyword_ids}})

keywords.find().count()

for doc in keywords.find():
    keywords.update_one({'_id': doc['_id']}, {'$set': {'frequency': len(doc['sentenceIDs'])} })

frequencies['islands']

for doc in papers.find({'paper': {'$exists': True}})[:5]:
    print(doc['authors'], doc['updated_keywords'])

papers = db.get_collection('papers')
sentences = db.get_collection('sentences')

a = [lambda doc: doc['_id'] for doc in sentences.find({}, {'_id': 1}).sort([('score',-1)]).limit(1)]

q = ['wrote', 'syntax']
answer = 'These are some papers you might want to read: \n\n'
for w in q:
    for candidate in papers.find({'updated_keywords': w}):
        answer += candidate['title'] + ', by '+ ', '.join(c for c in candidate['authors']) + '\n'
        answer += 'You can download the paper here: ling.auf.net/' + candidate['url'] + '\n\n'
print(answer)

a




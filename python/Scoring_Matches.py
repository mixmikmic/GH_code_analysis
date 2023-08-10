import pandas as pd
import sqlite3
import gensim
import nltk
import json
from gensim.corpora import BleiCorpus
from gensim import corpora
from nltk.corpus import stopwords
from textblob import TextBlob
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import numpy as np
import pickle
import glob

## Helpers

def save_pkl(target_object, filename):
    with open(filename, "wb") as file:
        pickle.dump(target_object, file)
        
def load_pkl(filename):
    return pickle.load(open(filename, "rb"))

def save_json(target_object, filename):
    with open(filename, 'w') as file:
        json.dump(target_object, file)
        
def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

model = LdaModel.load("aisnet_600_cleaned.ldamodel")
dictionary = Dictionary.load("aisnet_300_cleaned.ldamodel.dictionary")

# Helpers

def text2vec(text):
    if text:
        return dictionary.doc2bow(TextBlob(text.lower()).noun_phrases)
    else:
        return []
    
def tokenised2vec(tokenised):
    if tokenised:
        return dictionary.doc2bow(tokenised)
    else:
        return []
    
def predict(sometext):
    vec = text2vec(sometext)
    dtype = [('topic_id', int), ('confidence', float)]
    topics = np.array(model[vec], dtype=dtype)
    topics.sort(order="confidence")
#     for topic in topics[::-1]:
#         print("--------")
#         print(topic[1], topic[0])
#         print(model.print_topic(topic[0]))
    return pd.DataFrame(topics)

def predict_vec(vec):
    dtype = [('topic_id', int), ('confidence', float)]
    topics = np.array(model[tokenised2vec(vec)], dtype=dtype)
    topics.sort(order="confidence")
#     for topic in topics[::-1]:
#         print("--------")
#         print(topic[1], topic[0])
#         print(model.print_topic(topic[0]))
    return pd.DataFrame(topics)

def update_author_vector(vec, doc_vec):
    for topic_id, confidence in zip(doc_vec['topic_id'], doc_vec['confidence']):
        vec[topic_id] += confidence
    return vec

def get_topic_in_list(model, topic_id):
    return [term.strip().split('*') for term in model.print_topic(topic_id).split("+")]

def get_author_top_topics(author_id, top=10):
    author = authors_lib[author_id]
    top_topics = []
    for topic_id, confidence in enumerate(author):
        if confidence > 1:
            top_topics.append([topic_id, (confidence - 1) * 100])
    top_topics.sort(key=lambda tup: tup[1], reverse=True)
    return top_topics[:top]

def get_topic_in_string(model, topic_id, top=5):
    topic_list = get_topic_in_list(model, topic_id)
    topic_string = " / ".join([i[1] for i in topic_list][:top])
    return topic_string

def get_topics_in_string(model, topics, confidence=False):
    if confidence:
        topics_list = []
        for topic in topics:
            topic_map = {
                "topic_id": topic[0],
                "string": get_topic_in_string(model, topic[0]),
                "confidence": topic[1]
            }
            topics_list.append(topic_map)
    else:
        topics_list = []
        for topic_id in topics:
            topic_map = {
                "topic_id": topic_id,
                "string": get_topic_in_string(model, topic_id),
            }
            topics_list.append(topic_map)
    return topics_list

authors_lib = load_json("aisnet_600_cleaned.authors.json")

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

paper = np.array(authors_lib["1"]) - 1

paper = normalize(paper)

paper

def score(paper, scholar):
    return np.dot(paper, scholar) - 1

score(paper, authors_lib["2"])

scores = {}
for scholar, profile in authors_lib.items():
    scores[scholar] = score(paper, profile)

sorted_scores = [(k, scores[k]) for k in sorted(scores, key=scores.get, reverse=True)]

sorted_scores[:10]




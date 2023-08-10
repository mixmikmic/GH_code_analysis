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

con = sqlite3.connect("F:/FMR/data.sqlite")

db_documents = pd.read_sql_query("SELECT * from documents", con, index_col="id")
db_authors = pd.read_sql_query("SELECT * from authors", con)
len(db_documents)

db_documents.head()

tokenised = load_json("lemmatized.json")

non_en = load_pkl("non_en.list.pkl")

len(tokenised) == len(db_documents)

model = LdaModel.load("aisnet_600_cleaned.ldamodel")
dictionary = Dictionary.load("aisnet_300_cleaned.ldamodel.dictionary")

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

predict("null values are interpreted as unknown value or inapplicable value. This paper proposes a new approach for solving the unknown value problems with Implicit Predicate (IP). The IP serves as a descriptor corresponding to a set of the unknown values, thereby expressing the semantics of them. In this paper, we demonstrate that the IP is capable of (1) enhancing the semantic expressiveness of the unknown values, (2) entering incomplete information into database and (3) exploiting the information and a variety of inference rules in database to reduce the uncertainties of the unknown values.")

model.print_topic(167)

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

def profile_author(author_id, model_topics_num=None):
    if not model_topics_num:
        model_topics_num = model.num_topics
    author_vec = np.array([1.0 for i in range(model_topics_num)])
    # Initialize with 1s
    paper_list = pd.read_sql_query("SELECT * FROM documents_authors WHERE authors_id=" + str(author_id), con)['documents_id']
    paper_list = [i for i in paper_list if i not in non_en]
    # print(paper_list)
    for paper_id in paper_list:
        try:
            abstract = db_documents.loc[paper_id]["abstract"]
            vec = predict_vec(tokenised[paper_id -1])
        except:
            print("Error occurred on paper id " + str(paper_id))
            raise
        author_vec = update_author_vector(author_vec, vec)
    return list(author_vec) # to make it serializable by JSON

profile_author(1)

def profile_all_authors():
    authors = {}
    for author_id in db_authors['id']:
        result = profile_author(author_id)
        if len(result):
            authors[str(author_id)] = result # JSON does not allow int to be the key
        # print("Done: ", author_id)
        # uncomment the above line to track the progress
    return authors

authors_lib = profile_all_authors()

len(db_authors) == len(authors_lib)

save_json(authors_lib, "aisnet_600_cleaned.authors.json")


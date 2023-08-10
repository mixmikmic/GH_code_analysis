import json
import pandas as pd
import re
import random
from scipy import sparse
import numpy as np
from pymongo import MongoClient
from nltk.corpus import stopwords
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn_deltatfidf import DeltaTfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import sys
sys.path.append('/Users/robertsonwang/Desktop/Python/Yelp_class/yelp-classification/machine_learning')
import yelp_ml as yml
reload(yml)
from gensim import corpora, models, similarities, matutils
import tqdm

ip = '54.175.170.119'
conn = MongoClient(ip, 27017)
db = conn.get_database('cleaned_data')
reviews = db.get_collection('restaurant_reviews')
users_results = {}
user_id = []

lh_neg = open('../input/negative-words.txt', 'r').read()
lh_neg = lh_neg.split('\n')
lh_pos = open('../input/positive-words.txt', 'r').read()
lh_pos = lh_pos.split('\n')
users = json.load(open("cleaned_large_user_dictionary.json"))
word_list = list(set(lh_pos + lh_neg))

#Fix users JSON
users_dict = {}
user_ids = []

for list_reviews in users['reviews']:
    user_ids.append(list_reviews[0]['user_id'])
#We have 228 users, creat a new dictionary where the user_ids are the keys and the entries are a list of reviews
for i in tqdm.tqdm(range(0, len(user_ids))):
    users_dict[user_ids[i]] = users['reviews'][i]
with open('cleaned_large_user_dictionary.json', 'wb') as outfile:
    json.dump(users_dict, outfile)

#####Pull restaurant data for a given user
ip = '184.73.129.244'
conn = MongoClient(ip, 27017)
conn.database_names()
db = conn.get_database('cleaned_data')
reviews = db.get_collection('restaurant_reviews')

for user in users.keys()[0:1]:
    user_df = yml.make_user_df(users[user])
    business_ids = list(set(user_df['biz_id']))
#     restreview = {}
#     for i in tqdm.tqdm(range(0, len(business_ids))):
#         rlist = []
#         for obj in reviews.find({'business_id':business_ids[i]}):
#             rlist.append(obj)
#         restreview[business_ids[i]] = rlist
#     restaurant_df = yml.make_biz_df(user, restreview)
    
#     #Create a training and test sample from the user reviewed restaurants
#     split_samp = .20
#     random_int = random.randint(1, len(business_ids)-1)
#     len_random = int(len(business_ids) * split_samp)
#     test_set = business_ids[random_int:random_int+len_random]
#     training_set = business_ids[0:random_int]+business_ids[random_int+len_random:len(business_ids)]
#     train_reviews, train_ratings = [], []
    
    
#     #Create a list of training reviews and training ratings
#     for rest_id in training_set:
#         train_reviews.extend(list(user_df[user_df['biz_id'] == rest_id]['review_text']))
#         train_ratings.extend(list(user_df[user_df['biz_id'] == rest_id]['rating']))
    
#     #Transform the star labels into a binary class problem, 0 if rating is < 4 else 1
#     train_labels = [1 if x >=4 else 0 for x in train_ratings]
    
#     #Fit LSI model and return number of LSI topics
#     lsi, topics, dictionary = yml.fit_lsi(train_reviews)
    
#     #Make a FeatureUnion object with the desired features then fit to train reviews
#     comb_features = yml.make_featureunion(lda=False)
#     comb_features.fit(train_reviews)
    
#     train_features = comb_features.transform(train_reviews)
#     train_lsi = yml.get_lsi_features(train_reviews, lsi, topics, dictionary)
#     train_features = sparse.hstack((train_features, train_lsi))
#     train_features = train_features.todense()

    #Create a training and test sample from the user reviewed restaurants
    split_samp = .30
    random_int = random.randint(1, len(business_ids)-1)
    len_random = int(len(business_ids) * split_samp)
    test_set = business_ids[random_int:random_int+len_random]
    training_set = business_ids[0:random_int]+business_ids[random_int+len_random:len(business_ids)]
    sub_train_reviews, train_labels, train_reviews, train_ratings = [], [], [], []


    #Create a list of training reviews and training ratings
    for rest_id in training_set:
        train_reviews.append((user_df[user_df['biz_id'] == rest_id]['review_text'].iloc[0],
                                 user_df[user_df['biz_id'] == rest_id]['rating'].iloc[0]))

    #Create an even sample s.t. len(positive_reviews) = len(negative_reviews)
    sample_size = min(len([x[1] for x in train_reviews if x[1] < 4]),
                          len([x[1] for x in train_reviews if x[1] >= 4]))
    bad_reviews = [x for x in train_reviews if x[1] < 4]
    good_reviews = [x for x in train_reviews if x[1] >= 4]

    for i in range(0, int(float(sample_size)/float(2))):
        sub_train_reviews.append(bad_reviews[i][0])
        sub_train_reviews.append(good_reviews[i][0])
        train_labels.append(bad_reviews[i][1])
        train_labels.append(good_reviews[i][1])
    
    #Transform the star labels into a binary class problem, 0 if rating is < 4 else 1
    train_labels = [1 if x >=4 else 0 for x in train_labels]

    #Fit LSI model and return number of LSI topics
    lsi, topics, dictionary = yml.fit_lsi(sub_train_reviews)
    
    #Fit DeltaTFIDF Vecotrizer
    delta_vect = DeltaTfidfVectorizer(stop_words = 'english')
    delta_tfidf_vect = delta_vect.fit_transform(sub_train_reviews,train_labels)
    
    #Make a FeatureUnion object with the desired features then fit to train reviews
    comb_features = yml.make_featureunion()
    comb_features.fit(sub_train_reviews)

    train_features = comb_features.transform(sub_train_reviews)
    train_lsi = yml.get_lsi_features(sub_train_reviews, lsi, topics, dictionary)
    train_features = sparse.hstack((train_features, train_lsi, delta_tfidf_vect))
    train_features = train_features.todense()
    
    #fit each model in turn 
    model_runs = [(True, False, False), 
                  (False, True, False), 
                  (False, False, True)]
    test_results = {}
    for i in tqdm.tqdm(range(0, len(model_runs))):
        clf = yml.fit_model(train_features, train_labels, svm_clf = model_runs[i][0], 
                        RandomForest = model_runs[i][1], nb = model_runs[i][2])
        threshold = 0.7
        error = yml.test_user_set(test_set, clf, restaurant_df, user_df, comb_features, 
                                  threshold, lsi, topics, dictionary, delta_vect)
        test_results[clf] = error
    
    #Get scores
    for key in test_results.keys():
        results = test_results[key]
        log_loss = yml.get_log_loss(results)
        print "The log loss score is: " + str(log_loss)
        accuracy = yml.get_accuracy_score(results)
        print "The accuracy score is: " + str(accuracy)
        precision = yml.get_precision_score(results)
        print "The precision score is: " + str(precision)


"""
Created on Sun Feb  4 21:11:00 2018

@author: Pooja
www.poojaangurala.com
"""

import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

data = pd.read_csv('tweetreviews.csv') #tweets gathered in csv file
data1 = data[['tweets_only']] #extracting the col containing the tweets

all_pos_words = nltk.FreqDist(posTokens)
pos_word_features = list(all_pos_words.keys())
features_p = {}
def pos_tweet_token(tweets):
    t_words = set(tweets)
    for w in pos_word_features:
        features_p[w] = (w in t_words)

all_neg_words = nltk.FreqDist(negTokens)
neg_word_features = list(all_neg_words.keys())
features_n = {}
def neg_tweet_token(tweets):
    t_words = set(tweets)
    for w in neg_word_features:
        features_n[w] = (w in t_words)

def tweet_cat(data):
    cat=[]
    sent_token=[]
    for t in data['tweets_only']:
        sent_token.append(word_tokenize(t))
    for s in sent_token:
        words = set(s)
        stop_words = set(stopwords.words('English')) 
        clean_tweet= [i for i in words if not i in stop_words]
        clean_L_tweet=[]
        for i in clean_tweet:
            w=i.lower()
            clean_L_tweet.append(w)
        neg_tweet_token(clean_L_tweet)
        countn=0
        for fn, value in features_n.items():
            if value == True:
                countn=countn+1
        pos_tweet_token(clean_L_tweet)
        countp=0
        for key, value in features_p.items():
            if value == True:
                countp=countp+1
        if countp>countn:
            cat.append('pos')
        elif countp<countn:
            cat.append('neg')
        elif countp==countn:
            cat.append('nu')
    data['cat'] = cat
    return data   

tweet_cat(data1)

count_POS=0
count_NEG=0
count_NU=0
for i in data1['cat']:
    if i =='pos':
        count_POS=count_POS+1
    if i =='neg':
        count_NEG = count_NEG+1
    if i=='nu':
        count_NU = count_NU+1
print ("Total Pos: ",count_POS )
print ("Total Neg: ", count_NEG)
print ("Total Nu: ", count_NU)
print ("Total_count: ", len(data1['cat']))


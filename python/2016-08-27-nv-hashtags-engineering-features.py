import pymongo
import numpy as np
import pandas as pd
from collections import Counter
import re

client = pymongo.MongoClient()
db = client.sn_sp
coll = db.net_1

cursor = coll.find({})

items_list = [ item for item in cursor ] 

unique_hashstags = []

for tweet_doc in items_list:
    for h in tweet_doc['hashtags']:
        unique_hashstags.append(h)

unique_hashstags = list(set(unique_hashstags))

print len(unique_hashstags)
print unique_hashstags[0:20]

client = pymongo.MongoClient()
db = client.sn_sp
coll1 = db.net_1
coll2 = db.hashtags_1_count

for h in unique_hashstags:
    h_features = {'_id':h, 'sentiments':[], 'texts':[], 'retweet_counts':[]}
    cursor = coll1.find({'hashtags':h})
    for tweet_doc in cursor:
        h_features['sentiments'].append(tweet_doc['sentiment'])
        h_features['texts'].append(tweet_doc['text'])
        h_features['retweet_counts'].append(tweet_doc['retweet_count'])
    coll2.insert_one(h_features)

# NLP cleaning functions

def processTweet(tweet):
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert retweet
    tweet = re.sub('(rt\s)@[^\s]+','RETWEET',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet


def processWord(w):
    #look for 2 or more repetitions of character in a word and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    w = pattern.sub(r"\1\1", w)
    #strip punctuation
    w = w.strip('\'"?,.')
    #check if the word starts with an alphabet
    val = re.search(r"^[a-zA-Z][a-zA-Z0-9-]*$", w)
    if val is None:
        w = 'ABC'
    return w


def getStopWordList(stopWordListFileName):
    st = open(stopWordListFileName, 'r')
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')
    stopWords.append('RETWEET')
    stopWords.append('ABC')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords


def getWordsVector(tweet):
    # initialize vector
    wordsVector = []
    #initialize stopWords
    stopWords = getStopWordList('/Users/nicolavitale/Desktop/twitter_data_analysis/develop/data/SmartStoplist.txt')
    #process tweet and split into words
    tweet = processTweet(tweet)
    words = tweet.split()
    for w in words:
        w = processWord(w)
        if w in stopWords:
            continue
        else:
            wordsVector.append(w.lower())
    return wordsVector

################################################################################################################


def top10_words(statuses):
    words = []
    for s in statuses:
        wordsVector = getWordsVector(s)
        for word in wordsVector:
            words.append(word)
    c = Counter(words)
    return list(c.most_common()[:10])

    

def lexical_diversity(statuses):
    words = []
    for s in statuses:
        wordsVector = s.split()
        for word in wordsVector:
            words.append(word.lower())
    return 1.0*len(set(words))/len(words) 

def average_words(statuses):
    statuses = [s.lower() for s in statuses]
    total_words = sum([ len(s.split()) for s in statuses ])
    return 1.0*total_words/len(statuses)

def mean_sentiment(sentiments):
    sentiments_array = np.asarray(sentiments)
    return np.mean(sentiments_array)

def sentiment_variance(sentiments):
    sentiments_array = np.asarray(sentiments)
    return np.var(sentiments_array)

def count_positive(sentiments):
    return sentiments.count(1)

def count_negative(sentiments):
    return sentiments.count(0)

def mean_rtcount(retweet_counts):
    retweet_counts_array = np.asarray(retweet_counts)
    return np.mean(retweet_counts_array)

def rtcount_variance(retweet_counts):
    retweet_counts_array = np.asarray(retweet_counts)
    return np.var(retweet_counts_array)

client = pymongo.MongoClient()
db = client.sn_sp
coll1 = db.hashtags_1_count
coll2 = db.hashtags_2_count

cursor = coll1.find({})

for h in cursor:
    
    h_features = {'_id':h['_id'], 'top_10_words':[], 'lexical_diversity':0, 'average_words_n':0 , 'mean_sentiment':0, 'sentiment_variance':0, 'mean_rtcount':0, 'rtcount_variance':0}
    
    h_features['top_10_words']=top10_words(h['texts'])
    h_features['lexical_diversity']=lexical_diversity(h['texts'])
    h_features['average_words_n']=average_words(h['texts'])
    h_features['mean_sentiment']=mean_sentiment(h['sentiments'])
    h_features['sentiment_variance']=sentiment_variance(h['sentiments'])
    h_features['positive_count']=count_positive(h['sentiments'])
    h_features['negative_count']=count_negative(h['sentiments'])
    h_features['mean_rtcount']=mean_rtcount(h['retweet_counts'])
    h_features['rtcount_variance']=rtcount_variance(h['retweet_counts'])
    
    coll2.insert_one(h_features)

df = pd.read_csv('~/Desktop/twitter_data_analysis/develop/Py/tracked.csv') 
tracked_list = df['x'].tolist()

client = pymongo.MongoClient()
db = client.sn_sp
coll1 = db.hashtags_2_count
coll2 = db.hashtags_tracked_final_count

final_list = []

for h in tracked_list:
    tracked_h = coll1.find({"_id" : h})
    for i in tracked_h:
        final_list.append(i)

for i in final_list:
    coll2.insert_one(i)


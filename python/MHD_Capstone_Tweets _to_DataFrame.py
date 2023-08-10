import pandas as pd
import numpy as np 
import glob, os
import re
import sys
import io
from nltk import pos_tag
import nltk
import preprocessor as p
from sklearn.feature_extraction import text 

    
import warnings
warnings.filterwarnings('ignore')

tweet_dic = dict()
tweet_dic["tweet_id"] = []
tweet_dic["created_at"] = []
tweet_dic["text"] = []
tweet_dic["full_text"] = []
tweet_dic["user_id"] = []
tweet_dic["user_name"] = []
tweet_dic["user_screen_name"] = []
tweet_dic["hashtags"] = []
tweet_dic["mentions"] = []
tweet_dic["language"] = []
tweet_dic["source"] = []
tweet_dic["location"] = []

glob.glob("./data/untitled folder/*.txt")

# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# We use the file saved from last step as example

for f in glob.glob("./data/untitled folder/*.txt"): 
    
    tweets_file = open(f, "r")
    
    for line in tweets_file:
        try:
            # Read in one line of the file, convert it into a json object 
            tweet = json.loads(line.strip())

            if 'text' in tweet: 

                tweet_dic["tweet_id"].append(tweet["id"])

                tweet_dic["created_at"].append(tweet["created_at"])

                tweet_dic["text"].append(tweet["text"])

                try:
                    tweet_dic["full_text"].append(tweet["retweeted_status"]["extended_tweet"]["full_text"])
                except:
                    tweet_dic["full_text"].append("")

                tweet_dic["user_id"].append(tweet["user"]["id"])
                tweet_dic["user_name"].append(tweet["user"]["name"])
                tweet_dic["user_screen_name"].append(tweet["user"]["screen_name"])

                hashtags = []
                for hashtag in tweet["entities"]["hashtags"]:
                    hashtags.append(hashtag["text"])
                tweet_dic["hashtags"].append(hashtags)
            
                mentions_list = []
                for mention in tweet['entities']['user_mentions']:
                    mentions_list.append(mention["name"])
                    mentions_list.append(mention["screen_name"])
                tweet_dic["mentions"].append(mentions_list) 

                tweet_dic["language"].append(tweet["lang"])

                tweet_dic["source"].append(tweet["source"])

                tweet_dic["location"].append(tweet["coordinates"])

        except:
            # read in a line is not in JSON format (sometimes error occured)
            continue

tweet_df = pd.DataFrame(tweet_dic)

tweet_df.shape

tweet_df.head(3)

tweet_df = tweet_df.loc[tweet_df.language == "en", ]
tweet_df.shape

tweet_df.to_pickle("./unlabeled_tweets_df.pkl")

tweet_df2 = pd.read_pickle("./unlabeled_tweets_df.pkl")

tweet_df2.shape




import datetime
import pandas as pd
import requests
from blk import cfg
# https://github.com/Jefferson-Henrique/GetOldTweets-python
from get_old_tweets import got3 as got
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from requests_oauthlib import OAuth1
from pandas.io.json import json_normalize

def sentiment(df):
    tb = Blobber(analyzer=NaiveBayesAnalyzer())
    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['classification'] = df['text'].apply(lambda x: tb(x).sentiment.classification)
    df['p_pos'] = df['text'].apply(lambda x: tb(x).sentiment.p_pos)
    df['p_neg'] = df['text'].apply(lambda x: tb(x).sentiment.p_neg)

def query_hist(query, start_date, end_date):
    tweet_criteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(start_date).setUntil(end_date)
    tweets = got.manager.TweetManager.getTweets(tweet_criteria)

    for i in range(len(tweets)):
        d = {'index':tweets[i].date, 'text':tweets[i].text, 'id':tweets[i].id, 'username':tweets[i].username,
             'retweets':tweets[i].retweets, 'favorites':tweets[i].favorites,  'mentions':tweets[i].mentions,
             'hashtags':tweets[i].hashtags, 'geo':tweets[i].geo, 'permalink':tweets[i].permalink}
        if i == 0:
            df = pd.DataFrame.from_dict(d,orient='index').T
            df.index = df['index']
            df = df.drop('index', axis=1)
        else:
            df2 = pd.DataFrame.from_dict(d,orient='index').T
            df2.index = df2['index']
            df2 = df2.drop('index', axis=1)
            df = df.append(df2)
    sentiment(df)
    
    return df

query = '$eem'
start_date = '2017-01-03'
end_date = '2017-01-05'

df = query_hist(query, start_date, end_date)
df.head()

def username_lookup(un_list):
    df = pd.DataFrame()

    for i in range(0,len(un_list),100):
        working_list = un_list[i:i+100]
        usernames = ''
        for name in working_list:
            if name == working_list[-1]:
                usernames = usernames + name
                break
            else:
                string = name + '%2C'
                usernames = usernames + string

        url = 'https://api.twitter.com/1.1/users/lookup.json?screen_name=%s' % usernames
        auth = OAuth1(cfg.API_KEY, cfg.API_SECRET, cfg.ACCESS_TOKEN, cfg.ACCESS_TOKEN_SECRET)
        r = requests.get(url, auth=auth)
        for i in range(len(working_list)):
            try:
                name_df = json_normalize(r.json()[i])
                df = df.append(name_df)
            except:
                pass
    return df

un_list= df['username'].unique()
un_df = username_lookup(un_list)
un_df.head()

for c in un_df.columns:
    print(c)


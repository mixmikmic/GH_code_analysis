import requests
import numpy as np
import pandas as pd
import json
from IPython.display import display
import re
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

#from lib.twitter_keys import keys
get_ipython().system('pip install twitter tweepy')

CONSUMER_KEY = '0vqmq8vkBGwFaOOQJPkk75wj9'
CONSUMER_SECRET = 'F6FLN4514JXyq8IPmtIFQA9TI5TqJ3XFcxATrRhZ2GVQNDNhHf'
ACCESS_TOKEN = '862710988822372352-dC0d1LzeLd5wodTA3BqjRtT9U2f7TQR'
ACCESS_SECRET = 'J71AYBbH69nJCkqwPKV0YrqTDiUqjz0yOhNDJ9BGZ02kz'
keys = {
'CONSUMER_KEY': '0vqmq8vkBGwFaOOQJPkk75wj9',
'CONSUMER_SECRET': 'F6FLN4514JXyq8IPmtIFQA9TI5TqJ3XFcxATrRhZ2GVQNDNhHf',
'ACCESS_TOKEN': '862710988822372352-dC0d1LzeLd5wodTA3BqjRtT9U2f7TQR',
'ACCESS_SECRET': 'J71AYBbH69nJCkqwPKV0YrqTDiUqjz0yOhNDJ9BGZ02kz'
}

import requests
import numpy as np
import pandas as pd
import json
from IPython.display import display
import re
import os,sys,inspect
from time import sleep
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)


import json
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
CONSUMER_KEY = '0vqmq8vkBGwFaOOQJPkk75wj9'
CONSUMER_SECRET = 'F6FLN4514JXyq8IPmtIFQA9TI5TqJ3XFcxATrRhZ2GVQNDNhHf'
ACCESS_TOKEN = '862710988822372352-dC0d1LzeLd5wodTA3BqjRtT9U2f7TQR'
ACCESS_SECRET = 'J71AYBbH69nJCkqwPKV0YrqTDiUqjz0yOhNDJ9BGZ02kz'

### -----------------------------------------------------------------------------------####
### geo bounding for tweet location----------------------------------------------------####

los_angeles = "-118.670883,33.733477,-117.695847,34.290126"
Santa_Monica = "-118.514757,33.980857,-118.417253,34.065264"
Dallas = "-96.904907,32.761906,-96.684917,33.080035"
Midland_Odessa = "-103.1575,31.4849,-101.5178,32.3591"
Sacramento_east = "-121.8658,38.445,-120.2618,39.3598"
SFO = "-122.5319,37.5751,-122.3438,37.824"

### -----------------------------------------------------------------------------------####
### connect to postgres----------------------------------------------------------------####

import psycopg2 as pg2
import psycopg2.extras as pgex
this_host='34.211.59.66'
this_user='postgres'
this_password='postgres'


### ------------------------------------------------------------------------------------####
### cleaning text ----------------------------------------------------------------------####

def cleaner(text):
    text = text.lower()
    text = re.sub("'","''", text)
    text = re.sub("{","\{",text)
    text = re.sub("}","\}",text)
    text = re.sub('\n',' ',text)

    #text = re.sub(":","\:",text)
    return text

### -------------------------------------------------------------------------------------####
### cleaning tweet ----------------------------------------------------------------------####

#from spacy.en import STOP_WORDS
#from spacy.en import English
#import nltk
#nlp = English()
def tweet_cleaner(text):
    text = text.lower()
    text = re.sub("'","''", text)
    text = re.sub("{","\{",text)
    text = re.sub("}","\}",text)
    text = re.sub(r'http\S+', '',text)
    text = re.sub(r'@\S+', '',text)
    
    text = re.sub('\s+',' ',text)
    text = re.sub('\n',' ',text) 
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = re.sub(emoji_pattern, '', text)    
#    text = ' '.join([i.lemma_ for i in nlp(text) 
#                   if i.orth_ not in STOP_WORDS])
    
    return text

### -----------------------------------------------------------------------------------####
### Collecting tweets------------------------------------------------------------------####



oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter_stream = TwitterStream(auth=oauth)
#iterator = twitter_stream.statuses.filter(locations=Santa_Monica+','+\
#                                          Midland_Odessa+','+Dallas+','+\
#                                          Sacramento_east+','+SFO)
iterator = twitter_stream.statuses.filter(locations=los_angeles)
tweet_count = 300000

conn = pg2.connect(host = this_host, 
                        user = this_user,
                        password = this_password)


cur = conn.cursor()
for tweet in iterator:
    if tweet['lang'] == 'en':   
        tweet_count -= 1  

        try:
            id_str = str(tweet['id_str'])
        except:    
            pass
        try:
            screen_name = tweet['user']['screen_name']
        except:
            screen_name = None

        tweet_content = cleaner(tweet['text'])
        cleaned_tweet = tweet_cleaner(tweet['text'])
        date = tweet['created_at'][26:30]+'-'+tweet['created_at'][4:7]+'-'+tweet['created_at'][8:10]
        time = tweet['created_at'][11:19]
        
        screen_name = tweet['user']['screen_name']
        retweeted = tweet['retweeted']
        retweet_count = tweet['retweet_count']
        created_at = tweet['created_at']
        #date_time = "to_timestamp(concat(substring('{}',27,4),'-',substring('{}',5,3),'-',\
        #        substring'({}',9,2),' ',substring('{}',11,9)),\'YYYY-Mon-DD HH24:MI:SS') at time zone \'UTC'"\
        #        .format(created_at,created_at,created_at,created_at)
        date_time = tweet['created_at'][26:30]+'-'+                    tweet['created_at'][4:7]+'-'+tweet['created_at'][8:10]+' '+tweet['created_at'][11:19]
        get_hashtags = lambda tweet: " ".join([i for i in tweet.split() if ('#' in i)])
        hashtags1 = get_hashtags(tweet_content)
        hashtags1 = re.sub('\W',' ',hashtags1)
        hashtags1 = re.sub('\s+',' ',hashtags1)
        try: 
            if len(hashtags1) > 1:
                hashtags = hashtags1
            else:
                hashtags = None
        except:
            hashtags = None
        try:
            location =  cleaner(tweet['place']['full_name'])
        except:
            location = None
        try:
            country = tweet['place']['country']
        except:
            country = None
        try:
            place_type = tweet['place']['place_type']
        except:
            place_type = None
        try:
            latitude = tweet["geo"]["coordinates"][0]
            longitude = tweet["geo"]["coordinates"][1]
        except:
            latitude = 0.0 
            longitude = 0.0  
        usr = tweet['user']
        lang = tweet['lang']
        try:
            time_zone = cleaner(tweet['user']['time_zone'])
        except:
            time_zone = None    
        sql_insert = '''insert into tweets 
                            (
                                id,
                                screen_name,
                                tweet_content,
                                cleaned_tweet,
                                hashtags,
                                created_at,
                                date,
                                time,
                                date_time,
                                retweeted,
                                retweet_count,
                                location,
                                country,
                                place_type,
                                latitude,
                                longitude, 
                                time_zone,
                                lang
                            )
                        values
                            ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}');
                     '''.format(id_str,
                                screen_name,
                                tweet_content,
                                cleaned_tweet,
                                hashtags,
                                created_at,
                                date,
                                time,
                                date_time,
                                retweeted,
                                retweet_count,
                                location,
                                country,
                                place_type,
                                latitude,
                                longitude,
                                time_zone,
                                lang
                               )
        print(str(tweet_count)+' '+ screen_name+ ':  '+ tweet_content)
        #print(latitude,longitude)
        cur.execute(sql_insert)
            

        conn.commit()
        if tweet_count <= 0:
            break
    else:
        pass
conn.close()

tweet['created_at'][26:30]+'-'+tweet['created_at'][4:7]+'-'+tweet['created_at'][8:10]

tweet['created_at'][11:19]

import psycopg2 as pg2
import psycopg2.extras as pgex
this_host='54.191.217.176'
this_user='postgres'
this_password='postgres'
conn = pg2.connect(host = this_host, 
                        user = this_user,
                        password = this_password)
cur = conn.cursor(cursor_factory=pgex.RealDictCursor)
#cur.execute(sql_create)
#cur.execute(sql_drop)
#cur.execute(sql_insert)
#conn.commit()
cur.execute(sql_select)
rows = cur.fetchall()
conn.close()
df = pd.DataFrame(rows)

import requests
import numpy as np
import pandas as pd
import json
from IPython.display import display
import re
import os,sys,inspect
from time import sleep
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)


import json
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
CONSUMER_KEY = '0vqmq8vkBGwFaOOQJPkk75wj9'
CONSUMER_SECRET = 'F6FLN4514JXyq8IPmtIFQA9TI5TqJ3XFcxATrRhZ2GVQNDNhHf'
ACCESS_TOKEN = '862710988822372352-dC0d1LzeLd5wodTA3BqjRtT9U2f7TQR'
ACCESS_SECRET = 'J71AYBbH69nJCkqwPKV0YrqTDiUqjz0yOhNDJ9BGZ02kz'

### -----------------------------------------------------------------------------------####
### geo bounding for tweet location----------------------------------------------------####

los_angeles = "-118.670883,33.733477,-117.695847,34.290126"
Santa_Monica = "-118.514757,33.980857,-118.417253,34.065264"
Dallas = "-96.904907,32.761906,-96.684917,33.080035"
Midland_Odessa = "-103.1575,31.4849,-101.5178,32.3591"
Sacramento_east = "-121.8658,38.445,-120.2618,39.3598"
SFO = "-122.5319,37.5751,-122.3438,37.824"
[[-118.517358, 33.995177],
 [-118.517358, 34.050199],
 [-118.443482, 34.050199],
 [-118.443482, 33.995177]]
### -----------------------------------------------------------------------------------####
### connect to postgres----------------------------------------------------------------####

import psycopg2 as pg2
import psycopg2.extras as pgex
this_host='34.211.59.66'
this_user='postgres'
this_password='postgres'


### ------------------------------------------------------------------------------------####
### cleaning text ----------------------------------------------------------------------####

def cleaner(text):
    text = text.lower()
    text = re.sub("'","''", text)
    text = re.sub("{","\{",text)
    text = re.sub("}","\}",text)
    text = re.sub('\n',' ',text)

    #text = re.sub(":","\:",text)
    return text

### -------------------------------------------------------------------------------------####
### cleaning tweet ----------------------------------------------------------------------####

#from spacy.en import STOP_WORDS
#from spacy.en import English
#import nltk
#nlp = English()
def tweet_cleaner(text):
    text = text.lower()
    text = re.sub("'","''", text)
    text = re.sub("{","\{",text)
    text = re.sub("}","\}",text)
    text = re.sub(r'http\S+', '',text)
    text = re.sub(r'@\S+', '',text)
    
    text = re.sub('\s+',' ',text)
    text = re.sub('\n',' ',text) 
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = re.sub(emoji_pattern, '', text)    
#    text = ' '.join([i.lemma_ for i in nlp(text) 
#                   if i.orth_ not in STOP_WORDS])
    
    return text

### -----------------------------------------------------------------------------------####
### Collecting tweets------------------------------------------------------------------####



oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter_stream = TwitterStream(auth=oauth)
#iterator = twitter_stream.statuses.filter(locations=Santa_Monica+','+\
#                                          Midland_Odessa+','+Dallas+','+\
#                                          Sacramento_east+','+SFO)
iterator = twitter_stream.statuses.filter(locations=los_angeles)
tweet_count =2

conn = pg2.connect(host = this_host, 
                        user = this_user,
                        password = this_password)


cur = conn.cursor()
for tweet in iterator:
    
    if tweet['coordinates'] !='None': 
        tweet_count -= 1  
        print(tweet)
        print(tweet['coordinates'])
    if tweet_count <= 0:
        break   
conn.close()

tweet['place']['bounding_box']['coordinates'][0]

tweet['bounding_box']['coordinates']


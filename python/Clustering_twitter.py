import os
import sys
import json
import time
import math

from tweepy import API
from tweepy import OAuthHandler

from tweepy import Cursor
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

consumer_key    = 'xxxxxxxxxxxxxxxxxxxx'
consumer_secret = 'xxxxxxxxxxxxxxxxxxxx' 
access_token    = 'xxxxxxxxxxxxxxxxxxxx'
access_secret   = 'xxxxxxxxxxxxxxxxxxxx'

def get_twitter_auth():
    """Setup Twitter Authentication.
    
    Return: tweepy.OAuthHandler object
    """
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth
    
def get_twitter_client():
    """Setup Twitter API Client.
    
    Return: tweepy.API object
    """
    auth = get_twitter_auth()
    client = API(auth)
    return client

client = get_twitter_client()

screen_name = 'X1alejandro3x'

def paginate(items, n):
    """Generate n-sized chunks for items."""
    for i in range(0, len(items), n):
        yield items[i:i+n]

# Make directory
dirname = 'users/{}'.format(screen_name)
try:
    os.makedirs(dirname, mode=0o755, exist_ok=True)
except OSError:
    print('Directory {} already exists.'.format(dirname))
    
# Max num of requests per window
MAX_FRIENDS = 15000
max_pages = math.ceil(MAX_FRIENDS / 5000)
    
# get followers for a given user
fname = 'users/{}/followers.jsonl'.format(screen_name)
with open(fname, 'w') as f:
    for followers in Cursor(client.followers_ids, screen_name=screen_name).pages(max_pages):
        for chunk in paginate(followers, 100):
            users = client.lookup_users(user_ids=chunk)
            for user in users:
                f.write(json.dumps(user._json)+'\n')
        if len(followers) == 5000:
            print("More results available. Sleeping for 60 seconds to avoid rate limit")
            time.sleep(60)

k = 3                     # Number of clusters
max_features = 200        # Max number of features
max_ngram = 4             # Upper boundary for ngrams to be extracted

max_df = 0.8              # Max document freq for a feature 
min_df = 2                # Min document freq for a feature 
min_ngram = 1             # Lower boundary for ngrams to be extracted
use_idf = True            # True==TF-IDF, False==TF

with open(fname) as f:
    users = []
    for line in f:
        profile = json.loads(line)
        users.append(profile['description'])
        
    vectorizer = TfidfVectorizer(max_df=max_df,
                                 min_df=min_df,
                                 max_features=max_features,
                                 stop_words='english',
                                 ngram_range=(min_ngram, max_ngram),
                                 use_idf=use_idf)
    X = vectorizer.fit_transform(users)
    print('Data dimensions: {}'.format(X.shape))
    
    # perform clustering
    km = KMeans(n_clusters=k)
    km.fit(X)
    clusters = defaultdict(list)
    for i, label in enumerate(km.labels_):
        clusters[label].append(users[i])
        
    for label, descriptions in clusters.items():
        print('---------------------------------------------------- Cluster {}'.format(label))
        for desc in descriptions[:10]:
            print(desc)
        print('\n\n')


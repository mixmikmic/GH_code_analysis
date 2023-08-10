get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import warnings
import random
from datetime import datetime
random.seed(datetime.now())
warnings.filterwarnings('ignore')

# Make plots larger
plt.rcParams['figure.figsize'] = (10, 6)

# ACCESS_TOKEN = ''

ACCESS_TOKEN = 'EAACEdEose0cBABz7rh3DaV8cRCKU3iiQO04TKGGZCItk8AsBX7P9vatRMM88wp5H2ZBNmOuZBGcrLsRyZC4YSPA3kI6mB3D2gH3VlZA2s1rBGNZCiN7SPmJolKv7IW4R9FpvtZA6EfIqkY2A94BltOEJ82sQZC55rfIJBU9KC93iSMUBlmmgaJqtJruODLYQJRwZD'

import requests # pip install requests
import json

base_url = 'https://graph.facebook.com/me'

# Specify which fields to retrieve
fields = 'id,name,likes'

url = '{0}?fields={1}&access_token={2}'.format(base_url, fields, ACCESS_TOKEN)
print(url)

content = requests.get(url).json()
print(json.dumps(content, indent=1))

import facebook # pip install facebook-sdk

# Valid API versions are '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7'

# Create a connection to the Graph API with your access token
g = facebook.GraphAPI(ACCESS_TOKEN, version='2.7')

me=g.get_object('me')
print (me)
print (me['id'])

g.get_connections(id=me['id'], connection_name='posts')

g.get_connections(id=me['id'], connection_name='friends')

g.get_connections(id=me['id'], connection_name='feed')

# Get the active user's friends.
friends = g.get_connections(id=me['id'], connection_name='friends')
friends

# Search for a location
# Northeastern University  42.3398° N, 71.0892° W
g.request("search", {'type': 'place', 'center': '42.3398, -71.0892', 'fields': 'name, location'})

# Search for a user
g.request("search", {'q': 'Nik Bear Brown', 'type': 'user'})

# Search for a page
g.request("search", {'q': 'Deep Learning', 'type': 'page'})

# Search for a page
g.request("search", {'q': 'Blake Shelton', 'type': 'page'})

voice=['blakeshelton','MileyCyrus','jenniferhudson','OfficialAdamLevine']
feed = g.get_connections(voice[0], 'posts')
feed

def retrieve_page_feed(page_id, n_posts):
    """Retrieve the first n_posts from a page's feed in reverse
    chronological order."""
    feed = g.get_connections(page_id, 'posts')
    posts = []
    posts.extend(feed['data'])

    while len(posts) < n_posts:
        try:
            feed = requests.get(feed['paging']['next']).json()
            posts.extend(feed['data'])
        except KeyError:
            # When there are no more posts in the feed, break
            print('Reached end of feed.')
            break
            
    if len(posts) > n_posts:
        posts = posts[:n_posts]

    print('{} items retrieved from feed'.format(len(posts)))
    return posts

bs=retrieve_page_feed(voice[0], 33)
bs

bs[0]['id']

def fan_count(page_id):
    return int(g.get_object(id=page_id, fields=['fan_count'])['fan_count'])

bs_fc=fan_count(voice[0])
bs_fc

def post_engagement(post_id):
    likes = g.get_object(id=post_id, 
                         fields=['likes.limit(0).summary(true)'])\
                         ['likes']['summary']['total_count']
    shares = g.get_object(id=post_id, 
                         fields=['shares.limit(0).summary(true)'])\
                         ['shares']['count']
    comments = g.get_object(id=post_id, 
                         fields=['comments.limit(0).summary(true)'])\
                         ['comments']['summary']['total_count']
    return likes, shares, comments

engagement = post_engagement(bs[0]['id'])
engagement  # likes, shares, comments

def relative_engagement(e, total_fans):
    a=[]
    for i in e:
        a.append(i/total_fans)
    return a        

# Measure the relative share of a page's fans engaging with a post
re=relative_engagement(engagement,bs_fc)
re


# Define your unique twitter app credentials here
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""

get_ipython().magic('matplotlib inline')
import sys
sys.path.append('../')
from tweepy import OAuthHandler
from tweepy import Stream
from Twitter import Runtime
from Twitter import InMemory
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
import time
tweets = InMemory.Tweets()
listener = Runtime.Listener(tweets)
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, listener)

stream.filter(track=["lesbian"], async=True)

# Get our current tweets in the buffer
data = tweets.get_list()
raw = {}
# Loop over tweets and build our data sets
for created_at in data:
    # This tweet
    tweet = data[created_at]

    # Created At
    dt = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
    x = int(time.mktime(dt.timetuple()))
    
    # Number of URLs
    y = int(len(tweet['entities']['urls']))
    
    # X and Y values must b Integers
    plt.scatter(x,y)
    raw[x] = y
        
# Graph settings
plt.title("Tweets containing 'lesbian' - URLs over Epoch MS")
plt.xlabel('Epoch MS')
plt.ylabel('Number of URLs')

# Build the graph
plt.show()

# Comment this out to turn this on/off
for r in raw:
    x = r
    y = raw[x]
    print x, y



str = json.dumps(tweets.get_tweet(), indent=2, sort_keys=True)

# Comment this out to turn this on/off
#print str




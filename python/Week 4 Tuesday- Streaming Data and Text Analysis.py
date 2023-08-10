# Import Tweepy functions and include access keys and tokens in global namespace.

from tweet_stream import TwitterAuth, PrintStream, FileStream, get_stream

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# Create an OAuth object and make a connector

auth = TwitterAuth(consumer_key, consumer_secret, access_token, access_token_secret)
con = auth.make_connector()

# Set up listener and start stream with defined search terms.

listener = PrintStream()
stream = get_stream(con, listener)
stream.filter(track=[''])

# Check if the stream is still running

stream.running

# Disconnect the stream

stream.disconnect()

# Check to see if the stream is still active.

stream.running

# Set up listener and start stream with defined search terms.

listener = FileStream(filepath='')
stream = get_stream(con, listener)
stream.filter(track=[''])

import json

def tweets_list(filename):
    """
    Read lines from filepath and file into a list of dictionaries.
    
    Parameters
    ----------
    filename: str
    """
    tweets = []
    f = open(filename, 'r')
    for line in f:
        try:
            tweet = json.loads(line)
            tweets.append(tweet)
        except:
            continue
    return tweets     

# Read your tweet file with tweets_list without assigning to a variable (output to notebook)
tweets_list('')

# Assign the result of tweets_list to a variable 
data = tweets_list('')

# What is the type of the variable you assigned

# Explore tweet data structure starting with place in some specific cases

# Create dataframe from list of dicts
import pandas as pd

df = pd.DataFrame('')

# Examine the dataframe by looking at the columns. Horizontal scroll will be at the bottom if you call df directly.
df


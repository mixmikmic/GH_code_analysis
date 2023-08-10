import os
import tweepy
from textblob import TextBlob

consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")

access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

# authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# set access token
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# search for some tweets
public_tweets = api.search("Trump")

# loop through tweets and display their sentiments
for tweet in public_tweets:
    print (tweet.text)
    
    analysis = TextBlob(tweet.text)
    print (analysis.sentiment)




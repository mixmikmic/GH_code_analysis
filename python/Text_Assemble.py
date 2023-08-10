import textract

# Extracting from normal pdf
text = textract.process('Data/pdf/raw_text.pdf', language='eng')

# Extrcting from two columned pdf
text = textract.process('Data/pdf/two_column.pdf', language='eng')

# Extracting from scanned text pdf
text = textract.process('Data/pdf/ocr_text.pdf', method='tesseract', language='eng')

# Extracting from jpg
text = textract.process('Data/jpg/raw_text.jpg', method='tesseract', language='eng')
print text

text = textract.process('Data/wav/raw_text.wav', language='eng')
print "raw_text.wav: ", text

text = textract.process('Data/wav/standardized_text.wav', language='eng')
print "standardized_text.wav: ", text

from IPython.display import Image
Image(filename='../Chapter 5 Figures/Fetch_Twitter_Data.png', width=900)

#Import the necessary methods from tweepy library
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import pandas as pd

#provide your access details below 
access_token = "2345619938-zAlzAbKxL9aUqqqw02gCa6EWQWvoXVRVAVunS64"
access_token_secret = "LitbGHTzI0fmI76UsPWclvjVbKOzTt1G1jZqMwKu4CCW2"
consumer_key = "s3tHRlMAgThjVgscqJKL2o2vE"
consumer_secret = "1LTEr2CZtFXVzToOMx9ISd41F5PiAZSOzDHdspTyjnQlqjAQwJ"

# establish a connection
auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#fetch recent 10 tweets containing words iphone7 camera
fetched_tweets = api.search(q=['iPhone 7','iPhone7','camera'], result_type='recent', lang='en', count=10)
print "Number of tweets: ", len(fetched_tweets)

# Print the tweet text
for tweet in fetched_tweets:
    print 'Tweet ID: ', tweet.id
    print 'Tweet Text: ', tweet.text, '\n'

# function to save required basic tweets info to a dataframe
def populate_tweet_df(tweets):
    #Create an empty dataframe
    df = pd.DataFrame() 
    
    df['id'] = list(map(lambda tweet: tweet.id, tweets))
    df['text'] = list(map(lambda tweet: tweet.text, tweets))
    df['retweeted'] = list(map(lambda tweet: tweet.retweeted, tweets))
    df['place'] = list(map(lambda tweet: tweet.user.location, tweets))
    df['screen_name'] = list(map(lambda tweet: tweet.user.screen_name, tweets))
    df['verified_user'] = list(map(lambda tweet: tweet.user.verified, tweets))
    df['followers_count'] = list(map(lambda tweet: tweet.user.followers_count, tweets))
    df['friends_count'] = list(map(lambda tweet: tweet.user.friends_count, tweets))
    
    # Highly popular user's tweet could possibly seen by large audience, so lets check the popularity of user
    df['friendship_coeff'] = list(map(lambda tweet: float(tweet.user.followers_count)/float(tweet.user.friends_count), tweets))
    return df

df = populate_tweet_df(fetched_tweets) 
print df.head(10)

# For help about api look here http://tweepy.readthedocs.org/en/v2.3.0/api.html
fetched_tweets =  api.user_timeline(id='Iphone7review', count=5)

# Print the tweet text
for tweet in fetched_tweets:
    print 'Tweet ID: ', tweet.id
    print 'Tweet Text: ', tweet.text, '\n'


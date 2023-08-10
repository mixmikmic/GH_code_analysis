# Load keys, secrets, settings

import os

ENV = os.environ
CONSUMER_KEY = ENV.get('IOTX_CONSUMER_KEY')
CONSUMER_SECRET = ENV.get('IOTX_CONSUMER_SECRET')
ACCESS_TOKEN = ENV.get('IOTX_ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = ENV.get('IOTX_ACCESS_TOKEN_SECRET')
USERNAME = ENV.get('IOTX_USERNAME')
USER_ID = ENV.get('IOTX_USER_ID')

print(USERNAME)

import tweepy

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

public_tweets = api.home_timeline(count=3) 
for tweet in public_tweets:
    print(tweet.text)

# Models

user = api.get_user('clepy')
print(user.screen_name)
print(dir(user))

# Tweet!
status = api.update_status("I'm at @CLEPY!")
print(status.id)
print(status.text)

# Subclass StreamListener and define on_status method

class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print("@{0}: {1}".format(status.author.screen_name, status.text))

myStream = tweepy.Stream(auth = api.auth, listener=MyStreamListener())

try:
    myStream.filter(track=['#clepy'])
except KeyboardInterrupt:
    print('Interrupted...')
except tweepy.error.TweepError:
    myStream.disconnect()
    print('Disconnected. Try again!')




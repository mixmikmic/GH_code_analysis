import sys
import tweepy
import csv

# Step 1 - Authenticate
# pass security information to variables
consumer_key= ''
consumer_secret= ''

access_token=''
access_token_secret=''


# use variables to access twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


# Step 2 - Create an object called 'customStreamListener'

class CustomStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print (status.author.screen_name, status.created_at, status.text)
        # Writing status data
        with open('OutputStreaming.txt', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([status.author.screen_name, status.created_at, status.text])


    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

# Writing csv titles
with open('OutputStreaming.txt', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Author', 'Date', 'Text'])

streamingAPI = tweepy.streaming.Stream(auth, CustomStreamListener())
streamingAPI.filter(track=['data science', 'data scientist'])




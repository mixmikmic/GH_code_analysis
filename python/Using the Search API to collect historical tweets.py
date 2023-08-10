import jsonpickle
import tweepy

#Keys and tokens from the Twitter Application
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'
        
#Pass our consumer key and consumer secret to Tweepy's user authentication handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

#Pass our access token and access secret to Tweepy's user authentication handler
auth.set_access_token(access_token, access_token_secret)

#Creating a twitter API wrapper using tweepy
#Details here http://docs.tweepy.org/en/v3.5.0/api.html
api = tweepy.API(auth)
# if you use proxy, just add it to the API: proxy='https://your_proxy.server:port'

#Error handling
if (not api):
    print ("Problem connecting to API")

#Switching to application authentication
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

#Setting up new api wrapper, using authentication only
api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
 
#Error handling
if (not api):
    print ("Problem Connecting to API")

#You can check how many queries you have left using rate_limit_status() method
api.rate_limit_status()['resources']['search']

#This is what we are searching for
#We can search for multiple phrases using OR
searchQuery = '#Korea OR #PyeongChang2018'

#Maximum number of tweets we want to collect 
maxTweets = 3000

#The twitter Search API allows up to 100 tweets per query
tweetsPerQry = 100

tweetCount = 0

#Open a text file to save the tweets to
with open('tweets.json', 'w') as f:

    #Tell the Cursor method that we want to use the Search API (api.search)
    #Also tell Cursor our query, and the maximum number of tweets to return
    for tweet in tweepy.Cursor(api.search,q=searchQuery).items(maxTweets) :         
        
        #Write the JSON format to the text file, and add one to the number of tweets we've collected
        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
        tweetCount += 1

    #Display how many tweets we have collected
    print("Downloaded {0} tweets".format(tweetCount))


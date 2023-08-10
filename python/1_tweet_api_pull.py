import tweepy, cnfg, sys, os, logging, json

logging.basicConfig(filename='twitter_error_log.log',filemode='w', level=logging.ERROR)

config = cnfg.load(".twitter_config")
consumer_key = config['consumer_key']
consumer_secret = config['consumer_secret']

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
 
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True,
                 parser=tweepy.parsers.JSONParser()
                )
 
if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

# Twitter API search terms
climate_search = ('globalwarming OR (global AND warming) OR #climate OR (climate AND (science OR scientist OR scientists OR alarmist OR alarmists OR change OR realist OR realists OR denial OR denier OR deniers)) OR climatechange OR climaterealists OR #agw')
searchQuery = climate_search

maxTweets = 2000000000 # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits

# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are reqd, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1L

tweetCount = 0
tweetFiles = 0
msg =  "Downloading max {0} tweets".format(maxTweets)
logging.error(msg)
tweets = []
while 4000*tweetFiles + tweetCount < maxTweets:
    try:
        if (max_id <= 0):
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        since_id=sinceId)
        else:
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        max_id=str(max_id - 1))
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        max_id=str(max_id - 1),
                                        since_id=sinceId)
                
        new_tweets = new_tweets['statuses']
        if len(new_tweets) == 0:
            msg = 'No tweets found'
            logging.error(msg)
            break
        
        tweets.extend(new_tweets)
        tweetCount += len(new_tweets)
        msg = "Downloaded {0} tweets".format(tweetCount + tweetFiles*4000)
        logging.error(msg)
        max_id = new_tweets[-1]['id']
        
        if tweetCount > 4000:
            with open('tweets'+str(tweetFiles)+'.json', 'wb') as outfile:  
                json.dump(tweets, outfile)
            msg = "JSON file saved"
            logging.error(msg)
            tweetFiles += 1
            tweetCount = 0
            del tweets[:]
                
    except tweepy.TweepError as e:
        msg = 'Query failed when max_id equaled {0}: {1}'.format(max_id, e)
        logging.error(msg)

if tweetCount > 0:
    with open('tweets'+str(tweetFiles)+'.json', 'wb') as outfile:  
        json.dump(tweets, outfile)
    msg = "JSON file saved"
    logging.error(msg)
    tweetFiles += 1
    tweetCount = 0

with open('tweets0.json') as json_file:  
    test = json.load(json_file)

len(test)

test[0]

import requests
from application_only_auth import Client
client = Client(consumer_key, consumer_secret)

response = client.request(
    "https://api.twitter.com/1.1/application/rate_limit_status.json?resources=help,users,search,statuses")

response


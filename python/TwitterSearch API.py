from TwitterSearch import *

with open('token.txt','r') as f:
    token = f.read().split()

# pass your credentials to the TwitterSearch class to create and object called "ts"
ts = TwitterSearch(
consumer_key = token[0],
consumer_secret = token[1],
access_token = token[2],
access_token_secret = token[3]
)

tso = TwitterSearchOrder() # create a TwitterSearchOrder object
tso.set_keywords(['python']) # let's define all words we would like to have a look for
tso.set_language('en') # we want to see Enlish tweets only
tso.set_include_entities(False) # and don't give us all those entity information

# this is where the fun actually starts :)
for tweet in ts.search_tweets_iterable(tso):
    print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )

tweet_list = []
for tweet in ts.search_tweets_iterable(tso):
    tweet_list.append(tweet['text'])

len(tweet_list)

try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_keywords(['python', 'r']) # let's define all words we would like to have a look for
    tso.set_language('en') # we want to see English tweets only
    tso.set_include_entities(False) # and don't give us all those entity information

    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
            consumer_key = token[0],
            consumer_secret = token[1],
            access_token = token[2],
            access_token_secret = token[3]
        )

     # this is where the fun actually starts :)
    for tweet in ts.search_tweets_iterable(tso):
        print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )

except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)

try:
    tuo = TwitterUserOrder('wef') # create a TwitterUserOrder

    # it's about time to create TwitterSearch object again
    ts = TwitterSearch(
            consumer_key = token[0],
            consumer_secret = token[1],
            access_token = token[2],
            access_token_secret = token[3]
        )

    # start asking Twitter about the timeline
    for tweet in ts.search_tweets_iterable(tuo):
        print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )

except TwitterSearchException as e: # catch all those ugly errors
    print(e)


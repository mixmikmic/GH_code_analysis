get_ipython().system('pwd')

from os import chdir
chdir('/home/jovyan/work/Portfolio/Analyzing_Unstructured_Data_for_Finance/')

from lib import *
# suppress_warnings()
from lib.twitter_keys import my_keys

get_ipython().system('pip install pymongo pyquery tweepy')
import pymongo
import tweepy

# Replace the API_KEY and API_SECRET with your application's key and secret
auth = tweepy.AppAuthHandler(my_keys['CONSUMER_KEY'], my_keys['CONSUMER_SECRET'])

# Authorize twitter, initialize tweepy
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

# Identify port for better security of MongoDB
cli = pymongo.MongoClient(host='52.27.11.214', port=27016)

# Instantiates when you put data in 
task_collection = cli.twitter_db.task_collection
completed_collection = cli.twitter_db.completed_collection

cli.twitter_db.collection_names()

print(task_collection.count())
print(completed_collection.count())

# task_collection.drop()
# completed_collection.drop()

twitter_30_df = pd.read_csv('/home/jovyan/work/Portfolio/Analyzing_Unstructured_Data_for_Finance/data/1.twitter_users_30.csv', sep=',', encoding='latin-1')
handle = list(twitter_30_df['Handle'])

lookup_list = []
for h in handle:
    lookup_dict = {'handle': h,
                   'status': 'pending'}
    lookup_list.append(lookup_dict)

lookup_list

for dictionary in lookup_list:
    task_collection.insert_one(dictionary)

task_collection.count()

def get_tweets_to_mongo(lookup):

    maxTweets = 10000000 # Some arbitrary large number
    tweetsPerQry = 200  # this is the max the API permits

    # If results from a specific ID onwards are reqd, set since_id to that ID.
    # else default to no lower limit, go as far back as API allows
    sinceId = None

    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    max_id = -1E10

    tweetCount = 0
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.user_timeline(screen_name = lookup, count=tweetsPerQry)
                else:
                    new_tweets = api.user_timeline(screen_name = lookup, count=tweetsPerQry,
                                                    since_id=sinceId)
            else:

                if (not sinceId):
                    new_tweets = api.user_timeline(screen_name = lookup, count=tweetsPerQry,
                                                    max_id=str(max_id - 1))
                else:
                    new_tweets = api.user_timeline(screen_name = lookup, count=tweetsPerQry,
                                                    max_id=str(max_id - 1),
                                                    since_id=sinceId)
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                tweet_dict = {'username': tweet.user.screen_name,
                              'timestamp': tweet.created_at, 
                              'text': tweet.text.encode("utf-8")}
                mongo_response = completed_collection.insert_one(tweet_dict)

            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets for user: {1}".format(tweetCount, lookup))
            max_id = new_tweets[-1].id  

        except tweepy.TweepError as e:
            # Just exit if any error
            task_collection.insert_one({'handle': lookup, 'status': 'failed'})
            print("some error : " + str(e))
            break

    print ("Downloaded {0} tweets for user: {1} & saved to Mongo\n".format(tweetCount, lookup))

from datetime import datetime
start = datetime.now()

while True:
    # Get a user from the mongo collection
    task = task_collection.find_one_and_delete({'status': 'pending'})

    if type(task) != dict:
        print('Done!')
        break

    # Assign the handle to task_handle
    task_handle = task['handle']
        
        
    get_tweets_to_mongo(task_handle)
    task_collection.insert_one({'handle': task['handle'], 'status': 'complete'})
    
end = datetime.now()
print(end - start)

completed_collection.count()

# Look at the statuses in task_collection. Everything should be "complete."
curs = task_collection.find()
list_of_docs = []
for i in range(curs.count()):
    list_of_docs.append(curs.next())
    df = pd.DataFrame(list_of_docs)
df

# A faster way of getting data out of a collection
tweets_list = [document for document in completed_collection.find()]
tweets_df = pd.DataFrame(tweets_list)
tweets_df.head()

# Looks like we successfully got all our data from MongoDB
tweets_df.shape




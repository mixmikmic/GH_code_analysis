get_ipython().system('pip install --quiet pymongo')

get_ipython().system('pip install -U pymongo')

import pymongo
import pandas as pd
import pprint

client = pymongo.MongoClient('35.167.189.162', 27016)

client.database_names()

tweets_db = client.tweets

tweets_db

tweets_db.collection_names()

#tweets_db.my_gulf_collection.drop()

for a in tweets_db.my_eg_collection.find():
    pprint.pprint(a)

coll_ref_eg = tweets_db.my_eg_collection

coll_ref_eg.count()

cursor_eg = coll_ref_eg.find()

cursor_eg

eg_sample_docs = list(cursor_eg)

eg_tweets = list(coll_ref_eg.find())

eg_sample_docs[:5]

coll_ref_eg.count()

eg_tweets_df = pd.DataFrame(eg_sample_docs)

eg_tweets_df.info()

eg_tweets_df.sample(6)

eg_tweets_df['class'] = 'EG'

eg_tweets_df.sample(10)

for a in tweets_db.my_gulf_collection.find():
    pprint.pprint(a)

coll_ref_gulf = tweets_db.my_gulf_collection

coll_ref_gulf.count()

cursor_gulf = coll_ref_gulf.find()

cursor_gulf

gulf_sample_docs = list(cursor_gulf)

gulf_tweets = list(coll_ref_gulf.find())

gulf_sample_docs[:5]

gulf_tweets_df = pd.DataFrame(gulf_sample_docs)

gulf_tweets_df.info()

gulf_tweets_df['class'] = 'GULF'

gulf_tweets_df.head()

combined_df_streamed_timelines = eg_tweets_df.append([gulf_tweets_df])

combined_df_streamed_timelines.to_pickle('../CAPSTONE/Pickled_from_mongo/combined_eg_gulf.p')




#!pip install --quiet pymongo

import pymongo
import pandas as pd
import json
import pprint

client = pymongo.MongoClient('35.167.189.162', 27016)

client.database_names()

twitter_db = client.tweets

twitter_db.collection_names()

#twitter_db.my_gulf_collection.drop()

# for a in tweets_db.my_eg_collection.find():
#     pprint.pprint(a)

gulf_coll_ref = twitter_db.my_gulf_collection

client.database_names(), twitter_db.collection_names()

df_batch_1 = pd.read_pickle('../gulf_twitter_pickled/user_timeline_amoor99111.p')

df_batch_2 = pd.read_pickle('../gulf_twitter_pickled/user_timeline_alyazeedi96.p')

df_batch_3 = pd.read_pickle('../gulf_twitter_pickled/user_timeline_Basbosah.p')

#df_batch_4 = pd.read_pickle('../gulf_twitter_pickled/user_timeline_6al_asmaaa_.p')

df_batch_1.head(3)

df_batch_2.head(3)

df_batch_3.head(3)

#df_batch_4.head(3)

gulf_df = df_batch_1.append([df_batch_2, df_batch_3])

gulf_df.sample(10)

gulf_df.tail()

gulf_df.info()

gulf_coll_ref.count()

gulf_coll_ref.insert_many(gulf_df.to_dict('records'))

gulf_coll_ref.count()

client.database_names(), twitter_db.collection_names()





# df_json = df_batch_1.to_json()

# data = json.loads(df_json)

# coll_ref.insert_one(data)




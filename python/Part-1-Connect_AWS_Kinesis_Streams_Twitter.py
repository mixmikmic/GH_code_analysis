#Reading in twitter credentials
consumer_key = "XXXXXXXXXXXXXXXX"
consumer_secret = "XXXXXXXXXXXXXXXXXXX"
access_token_key = "XXXXXXXXXXXXXXx-XXXXXXXXXXXXXXXXXXXX"
access_token_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

#creating the kinesis stream
import boto3

client = boto3.client('kinesis')
response = client.create_stream(
   StreamName='twitter_bigdata', #your streamname here
   ShardCount=1
)

#importing the necessary packages
from TwitterAPI import TwitterAPI
import json
import boto3
#import twitterCreds


#accessing the API
api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret)

kinesis = boto3.client('kinesis')

r = api.request('statuses/filter', {'follow':'629747990'})

#for locations
#r = api.request('statuses/filter', {'locations':'-90,-90,90,90'})
#for userids @abcdef:
#r = api.request('statuses/filter', {'follow':'123456'})
#for general text searches
#r = api.request('statuses/filter', {'track':'iphone'})



for item in r:
    kinesis.put_record(StreamName="twitter_bigdata", Data=json.dumps(item), PartitionKey="filler")

import boto3
import time
import json
## aws creds are stored in ~/.boto
kinesis = boto3.client("kinesis")
shard_id = "shardId-000000000000" #only one shard!
pre_shard_it = kinesis.get_shard_iterator(StreamName="twitter", ShardId=shard_id, ShardIteratorType="LATEST")
shard_it = pre_shard_it["ShardIterator"]
while 1==1:
     out = kinesis.get_records(ShardIterator=shard_it, Limit=1)
     shard_it = out["NextShardIterator"]
     print out;
     time.sleep(1.0)


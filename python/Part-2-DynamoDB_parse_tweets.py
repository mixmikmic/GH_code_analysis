## create a table to store twitter hashtags in DynamoDB
import boto3
dynamodb = boto3.resource('dynamodb')
table = dynamodb.create_table(
    TableName='hashtags',
    KeySchema=[
        {
            'AttributeName': 'hashtag',
            'KeyType': 'HASH'
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'hashtag',
            'AttributeType': 'S'
        }
    ],
    # the pricing isdetermined by Provisioned Throughput, thus it is kept low
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)
table.meta.client.get_waiter('table_exists').wait(TableName='hashtags')

## Importing the necessary libraries

import time
import json
import decimal

#Connecting to the kinesis stream-need to specify kinesis stream here
kinesis = boto3.client("kinesis")
shard_id = 'shardId-000000000000' #only one shard
shard_it = kinesis.get_shard_iterator(StreamName="twitter_bigdata", ShardId=shard_id, ShardIteratorType="LATEST")["ShardIterator"]

#connecting to the dynamoDB table
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('hashtags')

#Parsing the tweets and setting a counter

while 1==1:
    out = kinesis.get_records(ShardIterator=shard_it, Limit=100)
    for record in out['Records']:
        if 'entities' in json.loads(record['Data']):
            htags = json.loads(record['Data'])['entities']['hashtags']
            if htags:
                    for ht in htags:
                        htag = ht['text']
                        checkItemExists = table.get_item(
                           Key={
                                        'hashtag':htag
                                }
                            )
                    if 'Item' in checkItemExists:
                        response = table.update_item(
                                Key={
                                    'hashtag': htag 
                            }, #updating the counter if hashtags exists
                            UpdateExpression="set htCount  = htCount + :val",
                            ConditionExpression="attribute_exists(hashtag)",
                            ExpressionAttributeValues={
                                ':val': decimal.Decimal(1) 	
                            },
                            ReturnValues="UPDATED_NEW"
                        )
                    else: #not updating the counter if hashtags not exist
                                        response = table.update_item(
                                                Key={
                                                        'hashtag': htag
                                                },
                                                UpdateExpression="set htCount = :val",
                                                ExpressionAttributeValues={
                                                        ':val': decimal.Decimal(1)
                                                },
                                                ReturnValues="UPDATED_NEW"
                                        )    
    shard_it = out["NextShardIterator"]
    time.sleep(1.0)




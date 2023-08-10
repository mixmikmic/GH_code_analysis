def sendPartition():
    pass

import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import pymongo
from pymongo import MongoClient

def sendRecord(record):

    word = record[0]
    count = record[1]
    print(word, count)
    
    client = MongoClient()
    db = client.fit5148_db
    collection = db.wc_coll
    collection.update({"_id": word}, {"$inc": {"count": count}}, upsert=True)
    client.close()
    
# We add this line to avoid an error : "Cannot run multiple SparkContexts at once". If there is an existing spark context, we will reuse it instead of creating a new context.
sc = SparkContext.getOrCreate()

# If there is no existing spark context, we now create a new context
if (sc is None):
    sc = SparkContext(appName="WordCountApp")
ssc = StreamingContext(sc, 2)
ssc.checkpoint("checkpoint")

host = "localhost"
port = 9999

lines = ssc.socketTextStream(host, int(port))

# Split each line into words
words = lines.flatMap(lambda line: line.split(" "))

# Count each word in each batch
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Output the result                            
wordCounts.foreachRDD(lambda rdd: rdd.foreach(sendRecord))

ssc.start()
try:
    ssc.awaitTermination(timeout=60)
except KeyboardInterrupt:
    ssc.stop()
    sc.stop()

ssc.stop()
sc.stop()

    


    




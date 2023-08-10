from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.functions import desc, col, window

from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
from pyspark.streaming import StreamingContext

import json
import time
import os

inputPath = "/home/ds/notebooks/datasets/lanl/"

numFiles = len(os.listdir(inputPath))
numFileOffset = numFiles - 1

print(f"There are {numFiles} files in our inputPath, which gives an offset of {numFileOffset}.")

APP_NAME = "Web Server Hypothesis Test"
SPARK_URL = "local[*]"

spark = SparkSession.builder.appName(APP_NAME).master(SPARK_URL).getOrCreate()

flowSchema = StructType([
    StructField('time', TimestampType(), True),
    StructField('duration', LongType(), True),
    StructField('srcdevice', StringType(), True),
    StructField('dstdevice', StringType(), True),
    StructField('protocol', LongType(), True),
    StructField('srcport', StringType(), True),
    StructField('dstport', StringType(), True),
    StructField('srcpackets', LongType(), True),
    StructField('dstpackets', LongType(), True),
    StructField('srcbytes', LongType(), True),
    StructField('dstbytes', LongType(), True)
])

# Static DataFrame representing data in the JSON files
staticInputDF = spark.read.json(inputPath)

staticInputDF.printSchema()

staticInputDF.select('dstdevice')     .where(col('dstport').isin([80, 443]))     .groupby('dstdevice')     .count()     .sort(desc('count'))     .show(20)

streamingInputDF = (
  spark
    .readStream                       
    .schema(flowSchema)               # Set the schema of the JSON data
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .json(inputPath)
)

streamingCountsDF = streamingInputDF     .select('dstdevice')     .where(col('dstport').isin([80, 443]))     .groupBy(streamingInputDF.dstdevice)     .count()     .sort(desc('count'))

# Is this DF actually a streaming DF?
streamingCountsDF.isStreaming

spark.conf.set("spark.sql.shuffle.partitions", "2")  # keep the size of shuffles small

query = (
  streamingCountsDF
    .writeStream
    .format("memory")       
    .queryName("counts")     # counts = name of the in-memory table
    .outputMode("complete")  # complete = all the counts should be in the table
    .start()
)

# let the query run for a bit to insure there is data in the recent progress structure.
time.sleep(4)

# Monitor the progress of the query. The last table should be identical to the static query.
while True:
    spark.sql("select * from counts").show(20)
    time.sleep(1)
    if query.recentProgress[-1]['sources'][0]['endOffset']['logOffset'] == numFileOffset:
        break




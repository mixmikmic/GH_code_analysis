import pyspark, pickle
from pyspark import SparkContext
from pyspark.sql.functions import countDistinct, regexp_replace, monotonically_increasing_id, lit
from pyspark.storagelevel import StorageLevel
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

spark = pyspark.sql.SparkSession.builder.getOrCreate()
sc = spark.sparkContext

tweets = spark.read.parquet('tweets_all.parquet')
tweets.persist(StorageLevel.MEMORY_AND_DISK);
tweets.count()

coms = spark.read.parquet('communities.parquet')
coms.persist(StorageLevel.MEMORY_AND_DISK)
coms.count()

pr = spark.read.parquet('pageranks.parquet')
pr.registerTempTable('pr')

tweets = tweets.join(coms, 'screen_name', 'inner')
tweets.registerTempTable('tweets')
coms.unpersist();

tweets = tweets.join(pr, on='screen_name', how='left')

tweets.filter('screen_name = "MrLukeyLuke"').select('text', 'created_at').take(10)

# Luke mostly communicated with other members of community 3, the climate change Deniers

tweets.filter('screen_name in ("ErikSolheim", "AdrianHarrop", "swinny198", "JSlate__", "Jack_P_95")').    select('screen_name', 'name', 'community','pagerank', 'retweeted_screen_name').show()




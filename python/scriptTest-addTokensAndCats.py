from pyspark.sql import SparkSession
from sentimentAnalysis import dataProcessing as dp

# create spark session
spark = SparkSession(sc)

# check dataframe wrote to s3
df = spark.read.json("s3a://amazon-review-data/review-data.json/test_data.json")

# subset asin, reviewText
df.show(3)

# add tokens
df_tokens = dp.add_tokens(df_subset)








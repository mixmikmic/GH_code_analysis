from pyspark.sql import SparkSession
from sentimentAnalysis import dataProcessing as dp

# create spark session
spark = SparkSession(sc)

# get dataframes
# specify s3 as sourc with s3a://
#df = spark.read.json("s3a://amazon-review-data/user_dedup.json.gz")
#df_meta = spark.read.json("s3a://amazon-review-data/metadata.json.gz")
df = spark.read.json("s3a://amazon-review-data/reviews_Musical_Instruments_5.json.gz")

# subset asin, reviewText
df_subset = df.select("asin", "reviewText")

# add tokens
df_tokens = dp.add_tokens(df_subset)

# add categories
df_cats = dp.add_categories(df_tokens, df_meta)

df_tokens.write.json("s3a://amazon-review-data/review-data.json")




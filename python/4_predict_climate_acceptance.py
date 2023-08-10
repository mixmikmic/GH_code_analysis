import pyspark, pickle
from pyspark import SparkContext
from pyspark.sql.functions import countDistinct, regexp_replace, monotonically_increasing_id
from pyspark.storagelevel import StorageLevel
import pandas as pd
import numpy as np
from pyspark.ml.feature import CountVectorizer, StringIndexer, StopWordsRemover, NGram, RegexTokenizer

from nltk.corpus import stopwords
import nltk, re

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline

from pyspark.sql.types import StringType

pd.options.display.max_colwidth = -1

spark = pyspark.sql.SparkSession.builder.getOrCreate()
sc = spark.sparkContext

tweets = spark.read.parquet('tweets_all.parquet')
tweets = tweets.orderBy('tweet_id').select('*', monotonically_increasing_id().alias('row'))
tweets.persist(StorageLevel.MEMORY_AND_DISK);

# Text needs to be renamed to 'tweet' for my model
# Links need to be replace with '[link]'
tweet_pipeline = tweets.select(regexp_replace('text', 'https?://[^ ,]+', '[link]').alias('tweet'))

nb_model = PipelineModel.load('./nb_model_pipeline/')

cc_accept_predictions = nb_model.transform(tweet_pipeline)
cc_accept_predictions.persist(StorageLevel.MEMORY_AND_DISK)
cc_accept_predictions = cc_accept_predictions.     select('probability', 'prediction', monotonically_increasing_id().alias('row'))

tweets = tweets.join(cc_accept_predictions, how='left',  on='row')
tweets.registerTempTable('tweets')

# 0 is 'accept,' 1 is 'deny'

spark.sql("""
    select screen_name, prediction, count(prediction)
    from tweets
    where screen_name = 'ClimateRealists'
    group by screen_name, prediction
    order by prediction
""").show(truncate=False)

spark.sql("""
    select screen_name, prediction, count(prediction)
    from tweets
    where screen_name = 'SteveSGoddard'
    group by screen_name, prediction
    order by prediction
""").show(truncate=False)

spark.sql("""
    select screen_name, prediction, count(prediction)
    from tweets
    where screen_name = 'ScottAdamsSays'
    group by screen_name, prediction
    order by prediction
""").show(truncate=False)

spark.sql("""
    select screen_name, prediction, count(prediction)
    from tweets
    where screen_name = 'JunkScience'
    group by screen_name, prediction
    order by prediction
""").show(truncate=False)

# 0 is 'accept,' 1 is 'deny'

spark.sql("""
    select screen_name, prediction, count(prediction)
    from tweets
    where screen_name = 'CoralMDavenport'
    group by screen_name, prediction
    order by prediction
""").show(truncate=False)

spark.sql("""
    select screen_name, prediction, count(prediction)
    from tweets
    where screen_name = 'NOAA'
    group by screen_name, prediction
    order by prediction
""").show(truncate=False)

spark.sql("""
    select screen_name, prediction, count(prediction)
    from tweets
    where screen_name = 'BillNye'
    group by screen_name, prediction
    order by prediction
""").show(truncate=False)

spark.sql("""
    select screen_name, prediction, count(prediction)
    from tweets
    where screen_name = 'EricHolthaus'
    group by screen_name, prediction
    order by prediction
""").show(truncate=False)

# Create dataframe with model vocabulary and weights for each word

theta = nb_model.stages[5].theta.toArray().transpose()
model_weights = pd.DataFrame(theta, index=nb_model.stages[4].vocabulary, columns=['accept', 'deny'])
model_weights.head()

# Calculate ratios for the weights of each word. Which ratios are furthest from 1?
model_weights.apply(lambda x: abs(1-(x.accept/x.deny)), axis=1).sort_values(ascending=False)

# Results look pretty good. The most predictive words look to be those used by climate change deniers.
# The  least predictive words could plausibly be used by either side.

# One problem with the model is that it can be very confident while being wrong
# EricHolthaus accepts climate change, yet the model sometimes predicts 1 ('deny') with high probability

spark.sql("""
    select screen_name, probability, prediction
    from tweets
    where screen_name = 'EricHolthaus' and prediction = 1
""").show(truncate=False)

# Another problem: many of the words in my tweets are not in my model vocabulary
# Take this tweet as an example, and output the words that are in the vocab:

t = []
for word in 'In which case most people are right. Not even the IPCC says climate change is "entirely" human-made'.split():
    if word in nb_model.stages[4].vocabulary:
        t.append(word)
print(t)

# Modal prediction. Not very good.

# predictions_per_user = spark.sql("""
#     select screen_name, min(prediction) as prediction
#     from
#         (select screen_name, prediction, count, 
#             rank() over (partition by screen_name order by count desc) as rank
#         from 
#             (select screen_name, prediction, count(*) as count
#             from tweets
#             group by screen_name, prediction) sub
#         ) sub2
#     where rank = 1
#     group by screen_name
# """)

# Ratio-based prediction. 'Acceptors' are those who have 4 times more 'accept' tweets than 'deny'

predictions_per_user = spark.sql("""
    select screen_name, case
                when n_denies = 0 then 'accept'
                when n_accepts/n_denies >= 4 then 'accept'
                else 'deny' end as prediction
    from
        (select screen_name,
            sum(case when prediction = 0 then 1 else 0 end) as n_accepts,
            sum(case when prediction = 1 then 1 else 0 end) as n_denies
        from tweets
        group by screen_name
        order by screen_name) sub
""")

# Note: these predictions didn't end up being good enough to be useful. I don't use them in my other final notebooks.

predictions_per_user.write.parquet('user_pred.parquet')

tweets.write.parquet('tweets_all_pred.parquet')


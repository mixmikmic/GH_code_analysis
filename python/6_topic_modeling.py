import pyspark, pickle
from pyspark import SparkContext
from pyspark.sql.functions import countDistinct, regexp_replace, monotonically_increasing_id, lit
from pyspark.storagelevel import StorageLevel
import pandas as pd
import numpy as np
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, RegexTokenizer
from pyspark.ml.clustering import LDA, LocalLDAModel

from nltk.corpus import stopwords
import nltk, re

from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

pd.options.display.max_colwidth = -1

spark = pyspark.sql.SparkSession.builder.getOrCreate()
sc = spark.sparkContext

def com_lda(df, community_number, k):
    """
    This function performs LDA on a given Twitter community with number of topics set to k.
    
    df: spark dataframe of tweet data
    community_number: ID number for a community
    k: number of topics
    returns: model vocabulary, count vectorized tweets, and model object
    """
    com_df = df.filter('community = {0}'.format(community_number))
    temp = pipeline.fit(com_df)
    vocab = temp.stages[2].vocabulary
    com_df_features = temp.transform(com_df)
    
    lda = LDA(k=k, maxIter=100, optimizer='online')
    model = lda.fit(com_df_features)
    
    return vocab, com_df_features, model

def print_top_words(model, vocab):
    """
    Prints the highest weighted words for each topic in a given LDA model.
    """
    top_words = model.describeTopics().rdd.map(lambda x: x['termIndices']).collect()
    
    for i, topic in enumerate(top_words):
        print('Topic {0}:'.format(i), end=' ')
        for index in topic:
            print(vocab[index], end=', ')
        print('\n')
        
def get_top_tweet_ids(model, df):
    """
    Uses the dot product of topic-by-word vectors and tweet-by-word-count vectors to score each tweet's
    relevance to an LDA topic.
    returns: 2D array of tweets for each LDA topic, sorted in order of relevance descending
    """
    ids = []
    m = model.topicsMatrix().toArray()
    n_topics = m.shape[1]
    for col in range(n_topics):
        topic = m[:, col]
        tweet_scores = df.rdd.map(lambda x: (x['features'].dot(topic), x['tweet_id'])).collect()
        ids.extend([tweet_id for score, tweet_id in sorted(tweet_scores, reverse=True)])
    res = np.array(ids).reshape(n_topics, -1)
    return res

tweets = spark.read.parquet('tweets_all.parquet')
coms = spark.read.parquet('communities.parquet')

# Remove URLs and @mentions from tweet text
# Use only tweets that were written originals, not retweets or quotes
lda_tweets = tweets.filter('retweet_id is null and quote_id is null').select('tweet_id', 'screen_name', 'name',
                  regexp_replace('text', r'(https?://[^ ,]+)|(@[\w_]+)', '').alias('text')) \
    .join(coms, 'screen_name', 'inner')
lda_tweets.persist(StorageLevel.MEMORY_AND_DISK)
lda_tweets.registerTempTable('lda_tweets')

lda_tweets.show()

# How many users per community?
spark.sql("""
    select community, count(distinct screen_name) as n_nodes
    from lda_tweets
    where community is not null
    group by community
    order by n_nodes desc
    limit 10
""").show()

tweet_stopwords = stopwords.words('english') +     ['rt', 'climate', 'change', 'global', 'warming', 'climatechange', 'climate', 'globalwarming', 'https', 'http',
        'amp', 'via', 'one', 'around', 'would', 'let', 'could', 'going', 'like', 'get', 'may', 'says', 'say', 'make',
        'based', 'even', 'another', 'completely', 'thanks', 'way', 'find', 'used', 'thing', '2017', 'see', 'need',
        'know', 'global-warming', 'climate-change', 'knows', 'think', 'thinks', 'take', 'new', 'day', 'days']

# Create regex tokenizer that is useful for Twitter data (preserves emoticons, hashtags, etc.)
# I used code from here, with some modifications: https://github.com/adonoho/TweetTokenizers/blob/master/PottsTweetTokenizer.py

pattern = r"""(?:\[link\])|(?:(?:\+?[01][\-\s.]*)?(?:[\(]?\d{3}[\-\s.\)]*)?\d{3}[\-\s.]*\d{4})|(?:(?<= )[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?)|(<[^>]+>)|(?:@[\w_]+)|(?:["\'][a-z0-9/-]+["\'])|(?:[a-z][a-z\-_]+[a-z])|(?:[+\-]?\d+[,/.:-]\d+[+\-]?)|(?:[\w_]+)"""

word_re = re.compile(pattern, re.VERBOSE | re.I | re.UNICODE)

# Tokenize tweets
tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", gaps=False, pattern=word_re.pattern,
                              minTokenLength = 2)

# Remove stopwords
stp_rmv = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='new_tokens',
                           stopWords=tweet_stopwords)

# Count occurences of words
cnvk = CountVectorizer(inputCol=stp_rmv.getOutputCol(), outputCol='features', vocabSize=10000)

# Pipeline
pipeline = Pipeline(stages=[tokenizer, stp_rmv, cnvk])

# Run LDA and save a list with all of the models

communities = [3,4,2,18,12,28]
vocab_list = []
df_features_list = []
model_list = []
ks = range(6,9)
for community in communities:
    for k in ks:
        vocab, com_df_features, model = com_lda(lda_tweets, community, k)
        vocab_list.append(vocab)
        df_features_list.append(com_df_features)
        model_list.append(model)

# Separate out Com 28 models
vocab28 = vocab_list[15:18]
df_features28 = df_features_list[15:18]
model28 = model_list[15:18]

# View top words for each k
for model, vocab in zip(model28, vocab28):
    print('Top Words for Community 12')
    print_top_words(model, vocab)
    print('\n')

# Get Com 28 k=7 top tweets
top28_7 = get_top_tweet_ids(model28[1], df_features28[1])

# Best tweets
lda_tweets.filter('tweet_id in {0}'.format(tuple(top28_7[3,:10]))).select('screen_name', 'community', 'text')                     .show(40, truncate=False)

# Save best Com 28 models
com28_model = model28[1]
com28_df = df_features28[1]
com28_vocab = vocab28[1]

com28_model.estimatedDocConcentration()

# Save all model data
com28_model.save('./lda_data/com28_lda_model') # Load with LocalLDAModel.load('lda_data/com28_lda_model')
with open('./lda_data/com28_lda_vocab.pkl', 'wb') as pklfile:
    pickle.dump(com28_vocab, pklfile)
com28_df.write.parquet('./lda_data/com28_df.parquet')

# Separate out Com 12 models
vocab12 = vocab_list[12:15]
df_features12 = df_features_list[12:15]
model12 = model_list[12:15]

# View top words for each k
for model, vocab in zip(model12, vocab12):
    print('Top Words for Community 12')
    print_top_words(model, vocab)
    print('\n')

# Get Com 12 k=8 top tweets
top12_8 = get_top_tweet_ids(model12[2], df_features12[2])

# Best tweets
lda_tweets.filter('tweet_id in {0}'.format(tuple(top12_8[7,:10]))).select('screen_name', 'community', 'text')                     .show(40, truncate=False)

# Save best Com 12 models
com12_model = model12[2]
com12_df = df_features12[2]
com12_vocab = vocab12[2]

com12_model.estimatedDocConcentration()

# Save all model data
com12_model.save('./lda_data/com12_lda_model')
with open('./lda_data/com12_lda_vocab.pkl', 'wb') as pklfile:
    pickle.dump(com12_vocab, pklfile)
com12_df.write.parquet('./lda_data/com12_df.parquet')

# Separate out Com 3 models
vocab3 = vocab_list[:3]
df_features3 = df_features_list[:3]
model3 = model_list[:3]

# View top words for each k
for model, vocab in zip(model3, vocab3):
    print('Top Words for Community 3')
    print_top_words(model, vocab)
    print('\n')

# Get Com 3 k=8 top tweets
top3_8 = get_top_tweet_ids(model_list[2], df_features_list[2])

# Save best Com 3 models
com3_model = model_list[2]
com3_df = df_features_list[2]
com3_vocab = vocab_list[2]

com3_model.estimatedDocConcentration()

# Save all model data
com3_model.save('./lda_data/com3_lda_model')
with open('./lda_data/com3_lda_vocab.pkl', 'wb') as pklfile:
    pickle.dump(com3_vocab, pklfile)
com3_df.write.parquet('./lda_data/com3_df.parquet')

# Separate out Com 4 models
vocab4 = vocab_list[3:6]
df_features4 = df_features_list[3:6]
model4 = model_list[3:6]

# View top words for each k
for model, vocab in zip(model4, vocab4):
    print('Top Words for Community 4')
    print_top_words(model, vocab)
    print('\n')

# Get Com 4 k=8 top tweets
top4_8 = get_top_tweet_ids(model4[2], df_features4[2])

# Best tweets
lda_tweets.filter('tweet_id in {0}'.format(tuple(top4_8[1,:10]))).select('screen_name', 'community', 'text')                     .show(40, truncate=False)

# Save best Com 4 models
com4_model = model4[2]
com4_df = df_features4[2]
com4_vocab = vocab4[2]

com4_model.estimatedDocConcentration()

# Save all model data
com4_model.save('./lda_data/com4_lda_model')
with open('./lda_data/com4_lda_vocab.pkl', 'wb') as pklfile:
    pickle.dump(com4_vocab, pklfile)
com4_df.write.parquet('./lda_data/com4_df.parquet')

# Separate out Com 18 models
vocab18 = vocab_list[9:12]
df_features18 = df_features_list[9:12]
model18 = model_list[9:12]

# View top words for each k
for model, vocab in zip(model18, vocab18):
    print('Top Words for Community 18')
    print_top_words(model, vocab)
    print('\n')

# Get Com 18 k=8
top18_8 = get_top_tweet_ids(model18[2], df_features18[2])

# Best tweets
lda_tweets.filter('tweet_id in {0}'.format(tuple(top18_8[2,:10]))).select('screen_name', 'community', 'text')                     .show(100, truncate=False)

# Save best Com 18 models
com18_model = model18[2]
com18_df = df_features18[2]
com18_vocab = vocab18[2]

# Save all model data
com18_model.save('./lda_data/com18_lda_model')
with open('./lda_data/com18_lda_vocab.pkl', 'wb') as pklfile:
    pickle.dump(com18_vocab, pklfile)
com18_df.write.parquet('./lda_data/com18_df.parquet')

# Separate out Com 2 models
vocab2 = vocab_list[6:9]
df_features2 = df_features_list[6:9]
model2 = model_list[6:9]

# View top words for each k
for model, vocab in zip(model2, vocab2):
    print('Top Words for Community 2')
    print_top_words(model, vocab)
    print('\n')

# Get Com 2 k=7
top2_7 = get_top_tweet_ids(model2[1], df_features2[1])

# Best tweets
lda_tweets.filter('tweet_id in {0}'.format(tuple(top2_7[1,:10]))).select('screen_name', 'community', 'text')                     .show(100, truncate=False)

# Save best Com 2 models
com2_model = model2[1]
com2_df = df_features2[1]
com2_vocab = vocab2[1]

# Save all model data
com2_model.save('./lda_data/com2_lda_model')
with open('./lda_data/com2_lda_vocab.pkl', 'wb') as pklfile:
    pickle.dump(com2_vocab, pklfile)
com2_df.write.parquet('./lda_data/com2_df.parquet')

# Run community 10
com10_vocab, com10_df, com10_model = com_lda(lda_tweets, 10, 10)

print_top_words(com10_model, com10_vocab)

top10_10 = get_top_tweet_ids(com10_model, com10_df)

# Best tweets 
lda_tweets.filter('tweet_id in {0}'.format(tuple(top_tweets[0,:20]))).select('screen_name', 'community', 'text')                     .show(truncate=False)

# Save all model data
com10_model.save('./lda_data/com10_lda_model')
with open('./lda_data/com10_lda_vocab.pkl', 'wb') as pklfile:
    pickle.dump(com10_vocab, pklfile)
com10_df.write.parquet('./lda_data/com10_df.parquet')

# Run community 9
community = 9
ks = range(6,9)
vocab9=[]
model9=[]
df_features9=[]
for k in ks:
    vocab, com_df_features, model = com_lda(lda_tweets, community, k)
    vocab9.append(vocab)
    df_features9.append(com_df_features)
    model9.append(model)
    print('Top Words for Community {0}, k = {1}'.format(community, k), '\n')
    print_top_words(model, vocab)
    print('\n')

# Get Com 9 k=8 top tweets
top9_8 = get_top_tweet_ids(model9[2], df_features9[2])

# Best tweets
lda_tweets.filter('tweet_id in {0}'.format(tuple(top9_8[1,:100]))).select('screen_name', 'community', 'text')                     .show(100, truncate=False)

# Save best Com 28 models
com9_model = model9[2]
com9_df = df_features9[2]
com9_vocab = vocab9[2]

com9_model.estimatedDocConcentration()

# Save all model data
com9_model.save('./lda_data/com9_lda_model')
with open('./lda_data/com9_lda_vocab.pkl', 'wb') as pklfile:
    pickle.dump(com9_vocab, pklfile)
com9_df.write.parquet('./lda_data/com9_df.parquet')

top_tweet_ids_save = [top10_10, top3_8, top4_8, top2_7, top18_8, top12_8, top28_7, top9_8]
n_keep_tweets = 100

 # Create empty DF to save to
top_tweets_df = lda_tweets.filter('tweet_id = "nonexistent id"').withColumn('topic', lit(None))

for com in top_tweet_ids_save:
    for topic_num in range(com.shape[0]):
        top_tweets_df = top_tweets_df.unionAll(
                                lda_tweets.filter('tweet_id in {0}'.format(tuple(com[topic_num,:n_keep_tweets]))) \
                                    .withColumn('topic', lit(topic_num)) \
                                    .select('screen_name', 'tweet_id', 'name', 'text', 'community', 'topic'))

top_tweets_df.persist(StorageLevel.MEMORY_AND_DISK)

top_tweets_df.show()

top_tweets_pd = top_tweets_df.toPandas()  

with open('./lda_data/lda_top_tweets.pkl', 'wb') as pklfile:
    pickle.dump(top_tweets_pd, pklfile, protocol=2)




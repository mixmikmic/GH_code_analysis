import re
from math import log
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("tweeter_spark").setMaster("local[2]")
sc   = SparkContext(conf=conf)

with open("../data/stop_words.txt", 'r') as f:
    STOPWORDS = f.read()
    
STOPWORDS = STOPWORDS.split("\n")
STOPWORDS = [word for word in STOPWORDS if word != '']

def prepTweets(file, label):
    """ preprocess the tweets by spliting each line into words, removing
    useless symbols, tranforming all words into lower cases, removing 
    label word and stop words.
    
    inputs:
    -------
    - file: the text file containing data.
    - label: the label of tweets inside this file.
    
    output:
    -------
    - tweets: the cleaned tweets dataset as an RDD. Each row contains the 
        cleaned words appeared in one tweet.
    - n_tweet: the number of tweets in this dataset.
    """
    # replace char ' with space
    # split the line into list of words by space
    # remove any characters are not alpha or number
    # change characters to lower case
    # remove label word
    # remove empty string
    # remove stop words
    tweets = sc.textFile(file)        .map(lambda x: x.replace("'", " "))        .map(lambda x: x.split(" "))        .map(lambda x: [re.sub(r'([^\s\w]|_)+', '', word) for word in x])        .map(lambda x: [word.lower() for word in x])        .map(lambda x: [word for word in x if word != label])        .map(lambda x: [word for word in x if word != ''])        .map(lambda x: [word for word in x if word not in STOPWORDS])
    return tweets, tweets.count()

resistTweets, n_resist = prepTweets("../data/resist.txt", 'resist')
magaTweets, n_maga     = prepTweets("../data/MAGA.txt", 'maga')
print("Number of resist tweets: %d" %n_resist)
print("Number of maga tweets: %d" %n_maga)

tweets   = resistTweets.union(magaTweets)
# flat the list so that each row has one word
# add 1 count to each word
# reduce them to get the word frequencies
# take the first 5000 words
featureCnt = tweets.flatMap(lambda x: x)    .map(lambda x: (x, 1))    .reduceByKey(lambda a, b: a+b)    .sortBy(lambda x: x[1], ascending=False)    .take(5000)
    
features, counts = zip(*featureCnt)
features[:10]

dataset_r = resistTweets.map(lambda x: [word in x for word in features])
dataset_m = magaTweets.map(lambda x: [word in x for word in features])

def elementWiseAdd(list_1, list_2):
    """ combine 2 lists together with element wise addition"""
    return [a + b for a, b in zip(list_1, list_2)]

wordCount_r = dataset_r.reduce(lambda a, b: elementWiseAdd(a, b))
wordCount_m = dataset_m.reduce(lambda a, b: elementWiseAdd(a, b))
wordCounts  = list(zip(features, wordCount_r, wordCount_m))

print("(word, word count in resist tweets, word count in maga tweets)")
wordCounts[:30]

def calcInf(cps):
    """ takes in the conditional probablities of one feature and calculate
    the informativeness of this feature.
    """
    return round(max(cps[0]/cps[1], cps[1]/cps[0]), 5)

# parallelize the word counts into rdd
# orgaize the rows as (word, (count_resist, count_maga))
# add one to each count to avoid zero divide error
# calculate the conditioning probabilities 
# calculate the informativeness
infs = sc.parallelize(wordCounts)    .map(lambda x: (x[0], (x[1], x[2])))    .mapValues(lambda x: (x[0]+1, x[1]+1))    .mapValues(lambda x: (x[0]/n_resist, x[1]/n_maga))    .mapValues(lambda x: (log(x[0]), log(x[1]), calcInf(x)))
    
# sort the features words by informativeness and collect them into master machine.
informativeness = infs    .sortBy(lambda x: x[1][2], ascending=False)    .collect()
    
infWords, _ = zip(*informativeness)
print("(word, logProb(word|label='resist'), logProb(word|label='maga'), informativeness)")
informativeness[:30]




import sys
print(sys.version)

spark

url = "s3n://spark-talk/reviews_Books_subset5.json"

review_subset = spark.read.json(url)

count = review_subset.count()
print("reviews_Books_subset5.json contains {} elements".format(count))

print("First 10 rows of review_subset DataFrame...")
review_subset.show(10, truncate=True)

import pyspark as ps    # for the pyspark suite
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
import string
import unicodedata
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser

from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF

import numpy as np
import sys


def extract_bow_from_raw_text(text_as_string):
    """Extracts bag-of-words from a raw text string.

    Parameters
    ----------
    text (str): a text document given as a string

    Returns
    -------
    list : the list of the tokens extracted and filtered from the text
    """
    if (text_as_string == None):
        return []

    if (len(text_as_string) < 1):
        return []

    import nltk
    if '/home/hadoop/nltk_data' not in nltk.data.path:
        nltk.data.path.append('/home/hadoop/nltk_data')

    nfkd_form = unicodedata.normalize('NFKD', unicode(text_as_string))
    text_input = nfkd_form.encode('ASCII', 'ignore')

    sent_tokens = sent_tokenize(text_input)

    tokens = map(word_tokenize, sent_tokens)

    sent_tags = map(pos_tag, tokens)

    grammar = r"""
        SENT: {<(J|N).*>}                # chunk sequences of proper nouns
    """

    cp = RegexpParser(grammar)
    ret_tokens = list()
    stemmer_snowball = SnowballStemmer('english')

    for sent in sent_tags:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'SENT':
                t_tokenlist = [tpos[0].lower() for tpos in subtree.leaves()]
                t_tokens_stemsnowball = map(stemmer_snowball.stem, t_tokenlist)
                #t_token = "-".join(t_tokens_stemsnowball)
                #ret_tokens.append(t_token)
                ret_tokens.extend(t_tokens_stemsnowball)
            #if subtree.label() == 'V2V': print(subtree)
    #tokens_lower = [map(string.lower, sent) for sent in tokens]

    stop_words = {'book', 'author', 'read', "'", 'character', ''}.union(ENGLISH_STOP_WORDS)

    tokens = [token for token in ret_tokens if token not in stop_words]

    return(tokens)


def indexing_pipeline(input_df, **kwargs):
    """ Runs a full text indexing pipeline on a collection of texts contained
    in a DataFrame.

    Parameters
    ----------
    input_df (DataFrame): a DataFrame that contains a field called 'text'

    Returns
    -------
    df : the same DataFrame with a column called 'features' for each document
    wordlist : the list of words in the vocabulary with their corresponding IDF
    """
    inputCol_ = kwargs.get("inputCol", "text")
    vocabSize_ = kwargs.get("vocabSize", 5000)
    minDF_ = kwargs.get("minDF", 2.0)

    tokenizer_udf = udf(extract_bow_from_raw_text, ArrayType(StringType()))
    df_tokens = input_df.withColumn("bow", tokenizer_udf(col(inputCol_)))

    cv = CountVectorizer(inputCol="bow", outputCol="vector_tf", vocabSize=vocabSize_, minDF=minDF_)
    cv_model = cv.fit(df_tokens)
    df_features_tf = cv_model.transform(df_tokens)

    idf = IDF(inputCol="vector_tf", outputCol="features")
    idfModel = idf.fit(df_features_tf)
    df_features = idfModel.transform(df_features_tf)

    return(df_features, cv_model.vocabulary)

review_df, vocab = indexing_pipeline(review_subset, inputCol='reviewText')

# Persist this DataFrame to keep it in memory
review_df.persist()

# print the top 5 elements of the DataFrame and schema to the log
print(review_df.take(5))
review_df.printSchema()

print("Example of first 50 words in our Vocab:")
print(vocab[:50])

from pyspark.ml.clustering import LDA

lda = LDA(k=5, maxIter=10, seed=42, featuresCol='features')
model = lda.fit(review_df)

import pandas as pd

model_description = model.describeTopics(20).toPandas()
vocab = np.array(vocab)

for idx, row in model_description.iterrows():
    desc = "Top Words Associated with Topic {0}:\n{1}\n"                 .format(row['topic'], vocab[row['termIndices']])
    print(desc)




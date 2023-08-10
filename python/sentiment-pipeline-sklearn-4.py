get_ipython().run_cell_magic('time', '', 'from fetch_twitter_data import fetch_the_data\nimport nltk\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n\ndf = fetch_the_data()\nX, y = df.text, df.sentiment\nX_train, X_test, y_train, y_test = train_test_split(X, y)\n\ntokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)\ncount_vect = CountVectorizer(tokenizer=tokenizer.tokenize) \nclassifier = LogisticRegression()')

def get_tweet_length(text):
    return len(text)

import numpy as np

def reshape_a_feature_column(series):
    return np.reshape(np.asarray(series), (len(series), 1))

def pipelinize_feature(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            processed = [function(i) for i in list_or_series]
            processed = reshape_a_feature_column(processed)
            return processed
#         This is incredibly stupid and hacky, but we need it to do a grid search with activation/deactivation.
#         If a feature is deactivated, we're going to just return a column of zeroes.
#         Zeroes shouldn't affect the regression, but other values may.
#         If you really want brownie points, consider pulling out that feature column later in the pipeline.
        else:
            return reshape_a_feature_column(np.zeros(len(list_or_series)))

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn_helpers import pipelinize, genericize_mentions, train_test_and_evaluate


sentiment_pipeline = Pipeline([
        ('genericize_mentions', pipelinize(genericize_mentions, active=True)),
        ('features', FeatureUnion([
                    ('vectorizer', count_vect),
                    ('post_length', pipelinize_feature(get_tweet_length, active=True))
                ])),
        ('classifier', classifier)
    ])

sentiment_pipeline, confusion_matrix = train_test_and_evaluate(sentiment_pipeline, X_train, y_train, X_test, y_test)


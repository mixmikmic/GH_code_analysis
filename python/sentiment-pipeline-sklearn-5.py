get_ipython().run_cell_magic('time', '', 'from fetch_twitter_data import fetch_the_data\nimport nltk\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n\ndf = fetch_the_data()\nX, y = df.text, df.sentiment\nX_train, X_test, y_train, y_test = train_test_split(X, y)\n\ntokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)\ncount_vect = CountVectorizer(tokenizer=tokenizer.tokenize) \nclassifier = LogisticRegression()')

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn_helpers import pipelinize, pipelinize_feature, get_tweet_length, genericize_mentions

sentiment_pipeline = Pipeline([
        ('genericize_mentions', pipelinize(genericize_mentions, active=True)),
        ('features', FeatureUnion([
                    ('vectorizer', count_vect),
                    ('post_length', pipelinize_feature(get_tweet_length, active=True))
                ])),
        ('classifier', classifier)
    ])

from sklearn.model_selection import GridSearchCV
from sklearn_helpers import train_test_and_evaluate
import numpy as np
import json

tokenizer_lowercase = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=False)
tokenizer_lowercase_reduce_len = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
tokenizer_uppercase = nltk.casual.TweetTokenizer(preserve_case=True, reduce_len=False)
tokenizer_uppercase_reduce_len = nltk.casual.TweetTokenizer(preserve_case=True, reduce_len=True)

# Our parameter dictionary
# You access parameters by giving the dictionary keys of <featurename>__<parameter>
# The values of each keys are a list of values that you want to test

parameters = {
    'genericize_mentions__kw_args': [{'active':False}, {'active':True}], # genericizing mentions on/off
    'features__post_length__kw_args': [{'active':False}, {'active':True}], # adding post length feature on/off
    'features__vectorizer__ngram_range': [(1,1), (1,2), (1,3)], # ngram range of tokenizer
    'features__vectorizer__tokenizer': [tokenizer_lowercase.tokenize, # differing parameters for the TweetTokenizer
                                        tokenizer_lowercase_reduce_len.tokenize,
                                        tokenizer_uppercase.tokenize,
                                        tokenizer_uppercase_reduce_len.tokenize,
                                        None], # None will use the default tokenizer
    'features__vectorizer__max_df': [0.25, 0.5], # maximum document frequency for the CountVectorizer
    'classifier__C': np.logspace(-2, 0, 3) # C value for the LogisticRegression
}

grid = GridSearchCV(sentiment_pipeline, parameters, verbose=1)

grid, confusion_matrix = train_test_and_evaluate(grid, X_train, y_train, X_test, y_test)

def print_best_params_dict(param_grid):
    used_cv = param_grid['features__vectorizer__tokenizer']
    if used_cv is None:
        params_to_print = grid.best_params_
        print 'used default CountVectorizer tokenizer'
    else:
        params_to_print = {i:grid.best_params_[i] for i in grid.best_params_ if i!='features__vectorizer__tokenizer'}
        print 'used CasualTokenizer with settings:'
        print '\tpreserve case: %s' % grid.best_params_['features__vectorizer__tokenizer'].im_self.preserve_case
        print '\treduce length: %s' % grid.best_params_['features__vectorizer__tokenizer'].im_self.reduce_len
    print 'best parameters: %s' % json.dumps(params_to_print, indent=4)
    
print_best_params_dict(grid.best_params_)


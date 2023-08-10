import pandas as pd
import numpy as np
df = pd.read_csv('singapore-roadnames-final-classified.csv')

# let's pick the same random 10% of the data to train with

import random
random.seed(1965)
train_test_set = df.loc[random.sample(df.index, int(len(df) / 10))]

X = train_test_set['road_name']
y = train_test_set['classification']

# our two ingredients: the ngram counter and the classifier
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,4), analyzer='char')

from sklearn.svm import LinearSVC
clf = LinearSVC()

from sklearn.pipeline import Pipeline, FeatureUnion

# There are just two steps to our process: extracting the ngrams and
# putting them through the classifier. So our Pipeline looks like this:

pipeline = Pipeline([
    ('vect', vect),  # extract ngrams from roadnames
    ('clf' , clf),   # feed the output through a classifier
])

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def run_experiment(X, y, pipeline, num_expts=100):
    scores = list()
    for i in range(num_expts):
        X_train, X_test, y_train, y_true = train_test_split(X, y)
        model = pipeline.fit(X_train, y_train)  # train the classifier
        y_test = model.predict(X_test)          # apply the model to the test data
        score = accuracy_score(y_test, y_true)  # compare the results to the gold standard
        scores.append(score)

    print sum(scores) / num_expts

# The general shape of a custom data transformer is as follows:

from sklearn.base import TransformerMixin, BaseEstimator

class DataTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, vars):
        self.vars = vars # this contains whatever variables you need 
                         # to pass in for use in the `transform` step
            
    def transform(self, data):
        # this is the crucial method. It takes in whatever data is passed into
        # the tranformer as a whole, such as a Pandas dataframe or a numpy array,
        # and returns the transformed data
        return mydatatransform(data, self.vars)
    
    def fit(self, *_):
        # most of the time, `fit` doesn't need to do anything
        # just return `self`
        # exceptions: if you're writing a custom classifier,
        #          or if how the test data is transformed is dependent on
        #                how the training data was transformed
        # Examples of the second type are scalers and the n-gram transformer
        return self

# Now let's actually write our extractor

class TextExtractor(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart 
       https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
       Also see Zac Stewart's excellent blogpost on pipelines:
       http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
       """
    
    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        # set the array type to be string
        return np.asarray(df[self.column_name]).astype(str)
        
    def fit(self, *_):
        return self

# Now let's update our previous code to operate on the full dataframe

random.seed(1965)
train_test_set = df.loc[random.sample(df.index, int(len(df) / 10))]

X = train_test_set[['road_name', 'has_malay_road_tag']]
y = train_test_set['classification']

pipeline = Pipeline([
    ('name_extractor', TextExtractor('road_name')), # extract names from df
    ('vect', vect),  # extract ngrams from roadnames
    ('clf' , clf),   # feed the output through a classifier
])

run_experiment(X, y, pipeline)

class Apply(BaseEstimator, TransformerMixin):
    """Applies a function f element-wise to the numpy array
    """
    
    def __init__(self, fn):
        self.fn = np.vectorize(fn)
        
    def transform(self, data):
        # note: reshaping is necessary because otherwise sklearn
        # interprets 1-d array as a single sample
        return self.fn(data.reshape(data.size, 1))

    def fit(self, *_):
        return self

# we already imported FeatureUnion earlier, so here goes

pipeline = Pipeline([
    ('name_extractor', TextExtractor('road_name')), # extract names from df
    ('text_features', FeatureUnion([
        ('vect', vect),  # extract ngrams from roadnames
        ('num_words', Apply(lambda s: len(s.split()))), # length of string
    ])),
    ('clf' , clf),   # feed the output through a classifier
])

run_experiment(X, y, pipeline)

# Okay! That didn't really improve our accuracy that much...let's try another feature

pipeline = Pipeline([
    ('name_extractor', TextExtractor('road_name')), # extract names from df
    ('text_features', FeatureUnion([
        ('vect', vect),  # extract ngrams from roadnames
        ('num_words', Apply(lambda s: len(s.split()))), # length of string
        ('ave_word_length', Apply(lambda s: np.mean([len(w) for w in s.split()]))), # average word length
    ])),
    ('clf' , clf),   # feed the output through a classifier
])

run_experiment(X, y, pipeline)

# That didn't help much either. Let's write another transformer that returns True
# if all the words in the roadname are in the dictionary
# we could use Apply and a lambda function for this, but let's be good and pass
# in the dictionary of words for better replicability

from operator import and_

class AllDictionaryWords(BaseEstimator, TransformerMixin):
    
    def __init__(self, dictloc='../resources/scowl-7.1/final/english-words*'):
        from glob import glob
        self.dictionary = dict()
        for dictfile in glob(dictloc):
            if dictfile.endswith('95'):
                continue
            with open(dictfile, 'r') as g:
                for line in g.readlines():
                    self.dictionary[line.strip()] = 1

        self.fn = np.vectorize(self.all_words_in_dict)
                
    def all_words_in_dict(self, s):
        return reduce(and_, [word.lower() in self.dictionary
                      for word in s.split()])

    def transform(self, data):
        # note: reshaping is necessary because otherwise sklearn
        # interprets 1-d array as a single sample
        return self.fn(data.reshape(data.size, 1))

    def fit(self, *_):
        return self

text_pipeline = Pipeline([
    ('name_extractor', TextExtractor('road_name')), # extract names from df
    ('text_features', FeatureUnion([
        ('vect', vect),  # extract ngrams from roadnames
        ('num_words', Apply(lambda s: len(s.split()))), # length of string
        ('ave_word_length', Apply(lambda s: np.mean([len(w) for w in s.split()]))), # average word length
        ('all_dictionary_words', AllDictionaryWords()),
    ])),
])

pipeline = Pipeline([
    ('text_pipeline', text_pipeline), # all text features
    ('clf' , clf),   # feed the output through a classifier
])

run_experiment(X, y, pipeline)

class BooleanExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        # set the array type to be string
        return np.asarray(df[self.column_name]).astype(np.bool)
                                                       
    def fit(self, *_):
        return self

malay_pipeline = Pipeline([
  ('malay_feature', BooleanExtractor('has_malay_road_tag')),
  ('identity', Apply(lambda x: x)), # this is a bit silly but we need to do the transform and this was the easiest way to do it
])

pipeline = Pipeline([
    ('all_features', FeatureUnion([
        ('text_pipeline', text_pipeline), # all text features
        ('malay_pipeline', malay_pipeline),
    ])),
    ('clf' , clf),   # feed the output through a classifier
])

run_experiment(X, y, pipeline)

from sklearn.pipeline import make_pipeline, make_union

def num_words(s):
    return len(s.split())

def ave_word_length(s):
    return np.mean([len(w) for w in s.split()])

def identity(s):
    return s

from sklearn.preprocessing import StandardScaler, MinMaxScaler

pipeline = make_pipeline(
    # features
    make_union(
        # text features
        make_pipeline(
            TextExtractor('road_name'),
            make_union(
                CountVectorizer(ngram_range=(1,4), analyzer='char'),
                make_pipeline(
                    Apply(num_words), # number of words
                    MinMaxScaler()
                ),
#                make_pipeline(
#                    Apply(ave_word_length), # average length of words
#                    StandardScaler()
#                ),
                AllDictionaryWords(),
            ),
        ),
        AveWordLengthExtractor(),
        # malay feature
        make_pipeline(
            BooleanExtractor('has_malay_road_tag'),
            Apply(identity),
        )
    ),
    # classifier
    LinearSVC(),
)

run_experiment(X, y, pipeline)


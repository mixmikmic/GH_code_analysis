import pandas as pd

dataset = pd.read_csv('yelp-review-subset.csv', header=0, delimiter=',', names=['stars', 'text', 'funny', 'useful', 'cool'])

# just checking the dataset
print('There are {0} star ratings, and {1} reviews'.format(len(dataset.stars.unique()), len(dataset)))
print(dataset.stars.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset[['text', 'funny', 'useful', 'cool']], dataset['stars'], train_size=0.8)

print(X_train.columns)

from sklearn.feature_extraction.text import CountVectorizer
# initialize a CountVectorizer
cv = CountVectorizer()
# fit the raw data into the vectorizer and tranform it into a series of arrays
X_train_counts = cv.fit_transform(X_train.text)
print(X_train_counts.shape)

# this is not what you want.
cv_test = CountVectorizer()
X_train_counts_test = cv.fit_transform(X_train)
print(X_train_counts_test.shape)

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report


class ItemSelector(TransformerMixin, BaseEstimator):
    """This class allows you to select a subset of a dataframe based on given column name(s)."""
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        return dataframe[self.keys]


class VotesToDictTransformer(TransformerMixin, BaseEstimator):
    """This tranformer converts the vote counts of each row into a dictionary."""
    def fit(self, x, y=None):
        return self
    
    def transform(self, votes):
        funny, useful, cool = votes['funny'], votes['useful'], votes['cool']
        return [{'funny': binarize_number(f, 1), 'useful': binarize_number(u, 1), 'cool': binarize_number(c, 1)} 
                for f, u, c in zip(funny, useful, cool)]

    
def binarize_number(num, threshold):
    return 0 if num < threshold else 1


pipeline = Pipeline([
    # Use FeatureUnion to combine the features from text and votes
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for getting BOW features from the texts
            ('bag-of-words', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('counts', CountVectorizer()),
            ])),

            # Pipeline for getting vote counts as features
            # the DictVecotrizer object there transform indexes the values of the dictionaries
            # passed down from the VotesToDictTransformer object.
            ('votes', Pipeline([
                ('selector', ItemSelector(keys=['funny', 'useful', 'cool'])),
                ('votes_to_dict', VotesToDictTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'bag-of-words': 1.0,
            'votes': 0.5
        },
    )),

    # Use a naive bayes classifier on the combined features
    ('clf', LogisticRegression()),
])


pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)
print(classification_report(predicted, y_test))

from textblob import TextBlob

class SentimentTransformer(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        features = []
        for text in texts:
            blob = TextBlob(text.decode('utf-8'))
            features.append({'polarity': binarize_number(blob.sentiment.polarity, 0.5),
                             'subjectivity': binarize_number(blob.sentiment.subjectivity, 0.5)})
        return features

pipeline = Pipeline([
        
    ('union', FeatureUnion(
        transformer_list=[
                    
            ('bag-of-words', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('counts', CountVectorizer()),
            ])),

            ('votes', Pipeline([
                ('selector', ItemSelector(keys=['funny', 'useful', 'cool'])),
                ('votes_to_dict', VotesToDictTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),
                    
            ('sentiments', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('sentiment_transform', SentimentTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),
        ],

        # weight components in FeatureUnion
        transformer_weights={
            'bag-of-words': 1.0,
            'votes': 0.5,
            'sentiments': 1.0,
        },
    )),

    # Use a naive bayes classifier on the combined features
    ('clf', LogisticRegression()),
])


pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)
print(classification_report(predicted, y_test))

from sklearn.feature_selection import SelectFromModel

pipeline = Pipeline([
        
    ('union', FeatureUnion(
        transformer_list=[
                    
            ('bag-of-words', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('counts', CountVectorizer()),
            ])),

            ('votes', Pipeline([
                ('selector', ItemSelector(keys=['funny', 'useful', 'cool'])),
                ('votes_to_dict', VotesToDictTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),
                    
            ('sentiments', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('sentiment_transform', SentimentTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),
        ],

        # weight components in FeatureUnion
        transformer_weights={
            'bag-of-words': 1.0,
            'votes': 0.5,
            'sentiments': 1.0,
        },
    )),

    # use SelectFromModel to select informative features
    ('feature_selection', SelectFromModel(LogisticRegression(C=0.5, penalty="l2"))),
    
    # Use a naive bayes classifier on the combined features
    ('clf', LogisticRegression()),
])


pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)
print(classification_report(predicted, y_test))

from sklearn.model_selection import GridSearchCV


pipeline = Pipeline([
        
    ('union', FeatureUnion(
        transformer_list=[
                    
            ('bag-of-words', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('counts', CountVectorizer()),
            ])),

            ('votes', Pipeline([
                ('selector', ItemSelector(keys=['funny', 'useful', 'cool'])),
                ('votes_to_dict', VotesToDictTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),
                    
            ('sentiments', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('sentiment_transform', SentimentTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),
        ],

        # weight components in FeatureUnion
        transformer_weights={
            'bag-of-words': 1.0,
            'votes': 0.5,
            'sentiments': 1.0,
        },
    )),

    # Use a naive bayes classifier on the combined features
    ('clf', LogisticRegression()),
])

params = dict(clf__max_iter=[50, 100, 150], clf__C=[1.0, 0.5, 0.1])
grid_search = GridSearchCV(pipeline, param_grid=params)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

pipeline = Pipeline([
        
    ('union', FeatureUnion(
        transformer_list=[
                    
            ('bag-of-words', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('counts', CountVectorizer()),
            ])),

            ('votes', Pipeline([
                ('selector', ItemSelector(keys=['funny', 'useful', 'cool'])),
                ('votes_to_dict', VotesToDictTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),
                    
            ('sentiments', Pipeline([
                ('selector', ItemSelector(keys='text')),
                ('sentiment_transform', SentimentTransformer()),
                ('vectorizer', DictVectorizer()),
            ])),
        ],

        # weight components in FeatureUnion
        transformer_weights={
            'bag-of-words': 1.0,
            'votes': 0.5,
            'sentiments': 1.0,
        },
    )),

    # Use a naive bayes classifier on the combined features
    ('clf', LogisticRegression(C=0.1, max_iter=50)),
])

pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)
print(classification_report(predicted, y_test))


get_ipython().system(' wget http://files.grouplens.org/datasets/movielens/ml-1m.zip -O ../data/ml-1m.zip')
get_ipython().system(' unzip ../data/ml-1m.zip -d ../data')

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

DATA_DIR = '../data/ml-1m/'
import os
ratings = (pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), 
                       engine='python', sep='::', names=['user', 'item', 'rating', 'timestamp'])
           .assign(timestamp=lambda df:pd.to_datetime(df.timestamp * 1000000000))
          )

movies = (pd.read_csv(os.path.join(DATA_DIR, 'movies.dat'), engine='python', sep='::', names=['item', 'title', 'genres'])
          .assign(genres=lambda df:df.genres.str.split('|').values)
          .set_index('item', drop=False))

# See http://files.grouplens.org/datasets/movielens/ml-1m-README.txt for more details
users = (
    pd.read_csv(os.path.join(DATA_DIR, 'users.dat'), engine='python', sep='::', 
                names=['user', 'gender', 'age', 'occupation', 'zipcode'])
    .set_index('user', drop=False))

from IPython.display import display, HTML

output_css = """
.output {
    flex-direction: column;
}
"""

HTML('<style>{}</style>'.format(CSS))

from sklearn import preprocessing
from itertools import chain

def columns_to_key_feature_pairs(row, key_column, feature_columns):
    return [(row[key_column], '{}={}'.format(column, row[column])) for column in feature_columns]

def array_column_to_key_feature_pairs(row, key_column, array_column):
    return [(row[key_column], u'{}={}'.format(array_column, value)) for value in row[array_column]]

feature_columns=['user', 'gender', 'occupation', 'zipcode']

user_features = pd.DataFrame.from_records(
    data=chain.from_iterable(
        columns_to_key_feature_pairs(row, key_column='user', feature_columns=feature_columns)
        for _, row in users.iterrows()),
    index='user',
    columns=['user', 'feature_name'])

item_features = pd.DataFrame.from_records(
    data=chain.from_iterable(
        columns_to_key_feature_pairs(row, key_column='item', feature_columns=['item']) +\
            array_column_to_key_feature_pairs(row, key_column='item', array_column='genres')
        for _, row in movies.iterrows()), 
    columns=['item', 'feature_name'],
    index='item')

features_encoder = preprocessing.LabelEncoder()
features_encoder.fit(np.hstack([user_features.feature_name, item_features.feature_name]))

user_features = user_features.assign(feature=lambda df: features_encoder.transform(df.feature_name))
item_features = item_features.assign(feature=lambda df: features_encoder.transform(df.feature_name))

display(user_features.head(10))
display(item_features.head(10))

batch_samples = ratings[['user', 'item', 'rating']].head(2)    .assign(sample_id=lambda df: np.arange(df.shape[0]))    .set_index('sample_id')

batch_samples_with_features = pd.concat([
    pd.merge(batch_samples, user_features, left_on='user', right_index=True),
    pd.merge(batch_samples, item_features, left_on='item', right_index=True)],
    axis=0).sort_index()

batch_samples_with_features

batch_samples_with_features    .groupby(by=batch_samples_with_features.index)    .feature.apply(np.array)

import scipy.sparse as sp

def to_sparse_indicators(featurized_batch_df):
    sample_ids_as_row_indexes = featurized_batch_df.index.values
    encoded_feature_as_col_indexes = featurized_batch_df.feature.values
    
    return sp.csr_matrix((
        np.ones_like(sample_ids_as_row_indexes),
                         (sample_ids_as_row_indexes, encoded_feature_as_col_indexes)))

batch_sparse = to_sparse_indicators(batch_samples_with_features)

batch_sparse

feature_type_to_range = pd.concat([user_features, item_features])    .assign(type=lambda df: df.feature_name.str.split('=').str[0])    .groupby('type').feature.aggregate([min, max])
    
feature_type_to_range

left_feature_start, left_feature_stop = feature_type_to_range.loc['gender'].values
right_feature_start, right_feature_stop = feature_type_to_range.loc['genres'].values

l = batch_sparse[:, left_feature_start:left_feature_stop+1]
r = batch_sparse[:, right_feature_start:right_feature_stop+1]

batch_interactions = sp.kron(l, r, format='csr')[:2:]

batch_interactions

from itertools import product

interaction_names = list(product(
    features_encoder.classes_[left_feature_start:left_feature_stop+1],
    features_encoder.classes_[right_feature_start:right_feature_stop+1]))

np.array(interaction_names)[batch_interactions[1].nonzero()[1]]




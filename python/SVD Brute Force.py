import os
import pandas as pd
import numpy as np
import datetime
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random

from surprise import Reader, Dataset, evaluate, print_perf, GridSearch
from surprise import SVD, BaselineOnly, Prediction, accuracy
from sklearn.metrics import roc_auc_score

random.seed(561)

# users = pd.read_csv('~/Columbia/Personalization Theory/lastfm-dataset-1K/userid-profile.tsv', header=None)
data = pd.read_csv('~/Columbia/Personalization Theory/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv',
                   delimiter="\t", header=None,
                   names = ["userid","timestamp","artistid",
                            "artistname","trackid","trackname"])

data['timestamp'] = pd.to_datetime(data['timestamp'])

data = data.groupby(['userid', 'artistname']).size().reset_index(name='plays')

users = list(np.sort(data.userid.unique())) # Get our unique users
artists = list(data.artistname.unique()) # Get our unique artists
quantity = list(data.plays) # All of our plays

rows = data.userid.astype('category', categories = users).cat.codes 
# Get the associated row indices
cols = data.artistname.astype('category', categories = artists).cat.codes 
# Get the associated column indices
plays_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(users), len(quantity)))
plays_sparse

# Sparsity of the matrix
matrix_size = plays_sparse.shape[0]*plays_sparse.shape[1] # Number of possible interactions in the matrix
num_plays = len(plays_sparse.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (num_plays/matrix_size))
sparsity

#rare_artists = data.query("plays < 6"). \
#    groupby('artistname').size().reset_index(name='users_listening_to_artist'). \
#    query("users_listening_to_artist < 10")
    
top5000_artists = data.groupby('artistname')['plays'].sum().reset_index(name='plays').     nlargest(5000,'plays')

reduced_data = data[data.artistname.isin(top5000_artists['artistname'])]

print(reduced_data.shape, data.shape)

users = list(np.sort(reduced_data.userid.unique())) # Get our unique users
artists = list(reduced_data.artistname.unique()) # Get our unique artists
quantity = list(reduced_data.plays) # All of our plays

rows = reduced_data.userid.astype('category', categories = users).cat.codes 
# Get the associated row indices
cols = reduced_data.artistname.astype('category', categories = artists).cat.codes 
# Get the associated column indices
plays_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(users), len(quantity)))

plays_sparse

# Sparsity of the matrix
matrix_size = plays_sparse.shape[0]*plays_sparse.shape[1] # Number of possible interactions in the matrix
num_plays = len(plays_sparse.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (num_plays/matrix_size))
sparsity

# free up memory
del users, artists, quantity, rows, cols, plays_sparse, matrix_size, num_plays, sparsity

# Normalize plays to be a percentage of total plays by artist
# usertotal = data.groupby('userid')['plays'].sum().reset_index(name="total_plays")
# normalized_data = pd.merge(reduced_data, usertotal)
# normalized_data['normalized_plays'] = normalized_data['plays']/normalized_data['total_plays']
# normalized_data.drop(['total_plays'], inplace=True, axis=1)
# normalized_data.loc[normalized_data['plays'] != 0, 'plays'] = 1

# set to binary of whether a user listed to an artist
data.loc[data['plays'] != 0, 'plays'] = 1
# remove all artists not in the top 5000
data = data[data.artistname.isin(top5000_artists['artistname'])]

# Add all user-artist combos, with no plays = 0
data = data.pivot(index='userid', columns='artistname', values='plays').fillna(0).reset_index()
data = data.melt(id_vars=['userid'], var_name=['artistname'])
data = data.rename(columns = {'value':'plays'})

print(len(data))
data.head()

reader = Reader(rating_scale=(0, 1))

# The columns must correspond to user id, item id and ratings (in that order).
model_data = Dataset.load_from_df(data, reader)
model_data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = BaselineOnly(bsl_options = {'method': 'als'})

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, model_data, measures=['RMSE', 'MAE'])

# predictions = predict(data['userid'], data['artistname'], data['plays'])
print_perf(perf)

reader = Reader(rating_scale=(0, 1))

param_grid = {'n_factors': np.arange(60,140,20),
              'lr_all': np.arange(0.002,0.014, 0.004),'reg_all': np.arange(0.02,0.1, 0.02)}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])

model_data = Dataset.load_from_df(data[['userid', 'artistname', 'plays']], reader)
model_data.split(n_folds=3)

grid_search.evaluate(model_data)

results_df = pd.DataFrame.from_dict(grid_search.cv_results)

#reader = Reader(rating_scale=(0, 1))

# The columns must correspond to user id, item id and ratings (in that order).
#model_data = Dataset.load_from_df(data, reader)
#model_data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD(n_factors = 120, lr_all = 0.01, reg_all = 0.02)

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, model_data, measures=['RMSE', 'MAE'])

# predictions = predict(data['userid'], data['artistname'], data['plays'])
print_perf(perf)

param_grid = {'n_factors': np.arange(60,140,8),'lr_all': [.01],'reg_all': [.02]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])

#model_data = Dataset.load_from_df(data[['userid', 'artistname', 'plays']], reader)
#model_data.split(n_folds=3)

grid_search.evaluate(model_data)

results_df = pd.DataFrame.from_dict(grid_search.cv_results)

results_df.to_csv('../data/factors.csv', sep='\t')

for trainset, testset in model_data.folds():

    # train and test algorithm.
    algo.train(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error and Mean Absolute Error
    #rmse = accuracy.rmse(predictions, verbose=True)
    #mae = accuracy.mae(predictions, verbose=True)

output = pd.DataFrame(predictions)
output = output.drop(['r_ui', 'details'], axis=1)

combined = pd.merge(data, output, how='right', left_on=['userid','artistname'], right_on=['uid','iid'])
combined = combined.drop(['uid', 'iid'], axis=1).set_index('userid')
combined['predicted'] = np.where(combined['est']>0.5, 1, 0)
combined['plays'] = np.where(np.isnan(combined['plays']),0,combined['plays'])
combined.describe()

#fpr, tpr, thresholds = metrics.roc_curve(combined['normalized_plays'], combined['est'], pos_label=2)
#metrics.auc(fpr, tpr)

roc_auc_score(combined['plays'],combined['est'])




import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
import time

# Load data
user_data_filepath = 'data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv'
user_profiles_filepath = 'data/lastfm-dataset-360K/usersha1-profile.tsv'

df_data = pd.read_table(user_data_filepath, header = None, 
                        names = ['user', 'musicbrainz-artist-id', 'artist', 'plays'],
                        usecols = ['user', 'artist', 'plays'])

df_user = pd.read_table(user_profiles_filepath,
                          header = None,
                          names = ['user', 'gender', 'age', 'country', 'signup'],
                          usecols = ['user', 'gender','country'])

len(df_data) #17535655
len(df_user) #359347

# Limit the data set to female users in United Kingdom only
df_user_UK = df_user[df_user['country']=='United Kingdom'].drop('country', axis=1)
df_user_UK_female = df_user_UK[df_user_UK['gender']=='f'].drop('gender', axis=1)
len(df_user_UK_female) #5851

# Merge the two dataframes
df = df_data.merge(df_user_UK_female, left_on='user', right_on ='user', how='inner')
df = df.groupby(['user','artist'], as_index=False).sum()
len(df) # 288780

# Find total number of plays for each artist
df_artist = df_data.groupby(['artist'])['plays'].sum().reset_index().rename(columns = {'plays': 'total_plays'})
df_artist.describe() 

# Find total number of plays of the 99th percentile artist
df_artist['total_plays'].quantile(0.99)  #198482.5899999995

# Set threshold = 200000 clicks, limit the data set to artists with more than 200000 clicks
df_top_artist = df_artist[df_artist['total_plays']>200000].sort('total_plays', ascending=False)
print("Top 10 artists: \n", df_top_artist[0:9])

top_artist = list(df_top_artist['artist'])
df = df[df['artist'].isin(top_artist)]
df.head()
len(df) #202917

# Create item-user matrix, where each row is the artist i (item) and each column is the user j,
## and the entry is the number of total plays clicked by user j to artist i.
matrix = df.pivot(index ='artist', columns='user', values='plays').fillna(0)
matrix_sparse = csr_matrix(matrix)

# checking
matrix.shape #(30593, 5841)   (2840, 5828)-top1%
matrix.index.get_loc('radiohead') #1976
matrix.index[1976]  #'radiohead'
matrix.iloc[1976]
matrix.loc["radiohead"] #same as above

item_similarity = pairwise_distances(matrix_sparse, metric='cosine')
user_similarity = pairwise_distances(matrix_sparse.T, metric='cosine')

# checking
item_similarity.shape #(2840, 2840)
user_similarity.shape #(5828, 5828)

# Make prediction
def predict(matrix, similarity, type='user'):
    if type == 'user':
        mean_user_rating = matrix.mean(axis=1)
        ratings_diff = (matrix - mean_user_rating)
        pred = mean_user_rating + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = matrix.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(matrix_sparse.T, item_similarity, type='item')
user_prediction = predict(matrix_sparse.T, user_similarity, type='user')

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic, evaluate

data = Dataset.load_from_df(df[['user', 'artist', 'plays']], Reader(rating_scale=(1, df['plays'].max())))

# First, train the algortihm to compute the similarities between items
# training is very slow..
trainset = data.build_full_trainset()

# compute cosine similarities between items
sim_options = {'name': 'cosine','user_based': False}
knn = KNNBasic(k=5, sim_options=sim_options)
knn.train(trainset)

# predict a certain item
userid = '000cdac4253f2dcecbd3dff8c5d7cf8cf0426c7a'
itemid = 'john mayer'
print(knn.predict(userid, itemid))

# actual rating 
print("Actual no. of plays of john mayer by user 0 =", matrix.loc["john mayer"][0])


ratings_data = "../../../data/ml-100k/u.data"
movies_data = "../../../data/ml-100k/u.item"

from collections import defaultdict

user_ratings = defaultdict(dict)
movie_ratings = defaultdict(dict)

with open(ratings_data, 'r') as f:
    for line in f:
        user, movie, stars, _ = line.split('\t')
        user_ratings[user][movie] = float(stars)
        movie_ratings[movie][user] = float(stars)

len(user_ratings)

len(movie_ratings)

user_ratings["1"]  # userID = 1

movies = {}
with open(movies_data, 'r', encoding="latin-1") as f:
    for line in f:
        movie_id, title, *_ = line.split('|')
        movies[movie_id] = title
        
len(movies)

movies["127"], movies["187"], movies["29"]  # movie ID = 127, 187, 29

movie_ratings["127"]

sum(movie_ratings["127"].values()) / len(movie_ratings["127"])

import pandas as pd
import numpy as np

ratings = pd.read_csv(ratings_data, sep='\t', names=['user', 'movie', 'rating', 'timestamp'])

ratings.head()

ratings.shape

n_movies = ratings["movie"].unique().shape
n_movies

n_users = ratings["user"].unique().shape
n_users

data_matrix = np.zeros((ratings.user.max(), ratings.movie.max()))

for item in ratings.itertuples():
    data_matrix[item.user-1, item.movie-1] = item.rating

data_matrix

data_matrix.shape

from scipy.spatial.distance import cosine

cosine(data_matrix[:, 126], data_matrix[:, 186])  # Godfather vs Godfather II

cosine(data_matrix[:, 126], data_matrix[:, 28])  # Godfather vs Batman Forever

cosine(data_matrix[0, :], data_matrix[2, :])  # user 1 vs user 3

cosine(data_matrix[0, :], data_matrix[915, :])  # user 1 vs user 916

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data_matrix, test_size=0.2)

train_data.shape, test_data.shape

from sklearn.metrics.pairwise import pairwise_distances

user_distance = pairwise_distances(train_data, metric='cosine')
item_distance = pairwise_distances(train_data.T, metric='cosine')

user_distance

user_similarity = 1 - user_distance
item_similarity = 1 - item_distance

user_similarity.shape, item_similarity.shape

train_data.shape

def make_user_prediction(data, u_similarity):
    return u_similarity.dot(data) / np.array([np.abs(u_similarity).sum(axis=1)]).T

def make_item_prediction(data, i_similarity):
    return data.dot(i_similarity) / np.array([np.abs(i_similarity).sum(axis=1)])

user_pred = make_user_prediction(train_data, user_similarity)
item_pred = make_item_prediction(train_data, item_similarity)

user_pred.shape

item_pred.shape

from sklearn.metrics import mean_squared_error

def matrix_mse(prediction, actual):
    prediction = prediction[actual.nonzero()].flatten()  # ignore zero terms
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(prediction, actual)

matrix_mse(user_pred, train_data)

matrix_mse(item_pred, train_data)




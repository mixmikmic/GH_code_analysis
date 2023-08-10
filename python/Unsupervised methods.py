# Import the libraries we will be using
import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 14, 8

np.random.seed(36)

# Read in the movies data
movies = pd.read_csv("data/movies.dat", names=['movie_id', 'movie_title', 'genre'], sep="\:\:")

# Movie ids don't start at 0 and some are missing, let's remap
movie_id_map = dict(zip(np.argsort(movies['movie_id'].unique())*-1, movies['movie_id'].unique()))

# Given the mapping, let's replace the values
movies = movies.replace({"movie_id": {v: k for k, v in movie_id_map.items()}})
movies['movie_id'] = movies['movie_id'] * -1

movies.head()

# Read in the ratings data
ratings = pd.read_csv("data/ratings.dat", names=['user_id', 'movie_id', 'rating', 'rating_timestamp'], sep="\:\:")

# User ids start at 1, let's bump them all down by 1
ratings['user_id'] = ratings['user_id'] - 1

# Make movie ids match the ones from our movie's data
ratings = ratings.replace({"movie_id": {v: k for k, v in movie_id_map.items()}})
ratings['movie_id'] = ratings['movie_id'] * -1

# Put our mapping back in order
movie_id_map = dict((key*-1, value) for (key, value) in movie_id_map.items())

ratings.head()

movies_ratings = pd.merge(movies, ratings, on="movie_id").drop(['genre', 'rating_timestamp'], axis=1)

movies_ratings.head()

data = sparse.csr_matrix((movies_ratings['rating'], (movies_ratings['movie_id'], movies_ratings['user_id'])), 
                         shape=(max(movies_ratings['movie_id'])+1, max(movies_ratings['user_id'])+1))

with open('data/movies_ratings.pickle', 'wb') as f:
    pickle.dump(data, f)

with open('data/movies_clean.pickle', 'wb') as f:
    pickle.dump(movies, f)

with open('data/movies_ratings.pickle', 'rb') as f:
    data = pickle.load(f)

with open('data/movies_clean.pickle', 'rb') as f:
    movies = pickle.load(f)

svd = TruncatedSVD(2)

svd.fit(data)

components = svd.transform(data)

plt.scatter(components[:,0], components[:,1])
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

def movie_search(movie_name):
    return movies.loc[(movies['movie_title'].str.match('(.*' + movie_name + '.*)').str.len() > 0)]

def movie_plotter(movie_id, components, x_buffer=3, y_buffer=2):
    x = components[movie_id][0]
    y = components[movie_id][1]

    xs = [x - x_buffer, x + x_buffer]
    ys = [y - y_buffer, y + y_buffer]

    plt.scatter(components[:,0], components[:,1])
    plt.xlim(xs)
    plt.ylim(ys)

    for x, y, title in zip(components[:,0], components[:,1], movies['movie_title']):
        if x >= xs[0] and x <= xs[1] and y >= ys[0] and y <= ys[1]:
            plt.text(x, y, title)

movie_search("The Matrix")

movie_plotter(7613, components, 4, 3)




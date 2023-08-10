import pandas as pd

movies = pd.read_csv("../data/intermediate/movies.csv", index_col=0)

movies.head()

import scipy.io

R = scipy.io.mmread("../data/intermediate/user_movie_ratings.mtx").tocsr()


print ('{0}x{1} user by movie matrix'.format(*R.shape))

from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(R.T)
print (similarities.shape)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

tsne = TSNE(perplexity=15, n_components=2, init="pca", n_iter=5000)
plot_only = 100
coords = tsne.fit_transform(similarities[:plot_only, :])

plt.figure(figsize=(18, 18))
labels = [movies.iloc[i].Title.decode("latin_1") for i in range(plot_only)]
for i, label in enumerate(labels):
    x, y = coords[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(10, 4),
                 textcoords="offset points",
                 ha="right",
                 va="bottom")

plt.show()

def similar_movies(model, movie_id, n=5):
    return model[movie_id].argsort()[::-1][:n].tolist()

movies.iloc[similar_movies(similarities, 4)]

import numpy as np

def predict_rating(model, ratings, movie_id, n=5):
    #
    # model = movie similarities matrix
    # movie_id = target movie id
    # ratings = dict of movie_id: rating
    # 
    
    rated_movies = ratings.keys()
    similar_movies = model[movie_id, rated_movies].argsort()[::-1]
    top_n = [ratings.keys()[i] for i in similar_movies[:n]]
    
    # Average rating weighted by similarity
    scores = sum(model[movie_id, m] * ratings[m] for m in top_n)
    
    prediction = float(scores) / sum(model[movie_id, m] for m in top_n)
    return prediction
    

user_id = 10
movies_rated = np.where(R[user_id].todense() > 0)[1].tolist()
movie_ratings = R[user_id, movies_rated].todense().tolist()[0]

user_ratings = dict(zip(movies_rated, movie_ratings))

for movie_id in [10, 100, 1000]:
    print movie_id, predict_rating(similarities, user_ratings, movie_id)

